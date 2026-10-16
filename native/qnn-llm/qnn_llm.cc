// libqnn-llm: универсальный рантайм LLM на NPU Qualcomm HTP (QNN) по metadata.json из qnn-llm-convert.
// Логика — порт converter/qnnllm/host.py: эмбеддинги на хосте (f16 или q4-блоки с zero point), аддитивные
// маски по позициям слотов (группа "context": слот = позиция, S = CL − N; кольцо W: слот = позиция mod W),
// RoPE из таблиц, KV-кэши с записью новых K/V после прогона всех частей, переносы между частями
// (carry: выход одной части напрямую в буфер входа следующей), рекуррентные состояния, логиты последней строки.
// C-API QnnLlm* — для P/Invoke из AiNpu и для qnn-llm-cli.
#include <algorithm>
#include <fstream>
#include <mutex>
#include <random>
#include <set>

#include "third_party/json.hpp"

#include "qnn_core.inc"  // открывает анонимное пространство имён (закрывается после Engine)

using json = nlohmann::json;

struct Mapped {
  void *p = nullptr;
  size_t size = 0;
  Mapped() = default;
  explicit Mapped(const std::string &path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) Die("open " + path);
    struct stat st;
    fstat(fd, &st);
    size = st.st_size;
    p = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) Die("mmap " + path);
  }
  Mapped(const Mapped &) = delete;
  Mapped &operator=(const Mapped &) = delete;
  ~Mapped() {
    if (p && p != MAP_FAILED) munmap(p, size);
  }
};

struct EmbeddingTable {
  int rows = 0, dim = 0, block = 32;
  float mult = 1.0f;
  bool is_f16 = true;
  std::set<int> remap;
  std::unique_ptr<Mapped> data, scales, zp;

  void Row(int id, f16 *out) const {
    if (remap.count(id)) id = 0;
    if (id < 0 || id >= rows) Die("токен вне словаря: " + std::to_string(id));
    if (is_f16) {
      const f16 *src = static_cast<const f16 *>(data->p) + (size_t)id * dim;
      for (int i = 0; i < dim; ++i) out[i] = (f16)((float)src[i] * mult);
      return;
    }
    const uint8_t *q = static_cast<const uint8_t *>(data->p) + (size_t)id * (dim / 2);
    int nb = dim / block;
    const f16 *s = static_cast<const f16 *>(scales->p) + (size_t)id * nb;
    const uint8_t *z = zp ? static_cast<const uint8_t *>(zp->p) + (size_t)id * ((nb + 1) / 2) : nullptr;
    for (int b = 0; b < nb; ++b) {
      float scale = (float)s[b] * mult;
      float zero = z ? (float)((b & 1) ? (z[b / 2] >> 4) : (z[b / 2] & 0x0F)) : 8.0f;
      for (int j = 0; j < block; ++j) {
        int k = b * block + j;
        int v = (k & 1) ? (q[k / 2] >> 4) : (q[k / 2] & 0x0F);
        out[k] = (f16)(((float)v - zero) * scale);
      }
    }
  }
};

struct CacheSpec {
  std::string group;
  bool bhdt = true;
  int heads = 0, dim = 0;
};

struct Group {
  bool context = true;
  int ring = 0;
  std::vector<int64_t> slot_pos;
};

class Engine {
 public:
  Engine(const std::string &lib_dir, const std::string &model_dir) {
    std::ifstream f(model_dir + "/metadata.json");
    if (!f) Die("нет " + model_dir + "/metadata.json");
    meta_ = json::parse(f);
    cl_ = meta_["context_length"];
    for (int n : meta_["ar"]) ars_.push_back(n);
    ar_ = *std::max_element(ars_.begin(), ars_.end());
    qnn_ = std::make_unique<Qnn>(lib_dir + "/libQnnHtp.so", lib_dir + "/libQnnSystem.so");
    for (auto &e : meta_["embeddings"]) {
      auto t = std::make_unique<EmbeddingTable>();
      t->rows = e["rows"];
      t->dim = e["dim"];
      t->mult = e.value("multiplier", 1.0f);
      t->is_f16 = e["format"] == "f16";
      t->block = e.value("block_size", 32);
      for (int id : e.value("remap_to_zero", std::vector<int>{})) t->remap.insert(id);
      t->data = std::make_unique<Mapped>(model_dir + "/" + e["file"].get<std::string>());
      if (!t->is_f16) {
        t->scales = std::make_unique<Mapped>(model_dir + "/" + e["scales"].get<std::string>());
        if (e.contains("zero_points")) t->zp = std::make_unique<Mapped>(model_dir + "/" + e["zero_points"].get<std::string>());
      }
      emb_[e["name"]] = std::move(t);
    }
    for (auto &r : meta_["rope"]) {
      int pos = r["positions"], half = r["half_dim"];
      Mapped m(model_dir + "/" + r["file"].get<std::string>());
      const float *src = static_cast<const float *>(m.p);
      auto &tab = rope_[r["name"]];
      tab.half = half;
      tab.cos.assign(src, src + (size_t)pos * half);
      tab.sin.assign(src + (size_t)pos * half, src + (size_t)2 * pos * half);
    }
    for (auto &g : meta_["cache_groups"]) {
      Group grp;
      grp.context = g["slots"].is_string();
      grp.ring = grp.context ? 0 : g["slots"].get<int>();
      grp.slot_pos.assign(grp.context ? cl_ : grp.ring, -1);
      groups_[g["name"]] = grp;
    }
    for (auto &c : meta_["caches"]) caches_[c["name"]] = {c["group"], c["layout"] == "BHDT", c["heads"], c["dim"]};
    for (auto &s : meta_.value("states", json::array())) {
      size_t elems = 1;
      for (int d : s["shape"]) elems *= d;
      state_[s["name"]].Alloc(elems);
    }
    for (auto &p : meta_["parts"]) {
      for (auto &g : qnn_->Load(model_dir + "/" + p["file"].get<std::string>())) graphs_[g->name] = std::move(g);
    }
    // Сначала графы decode (меньший N): их KV-кэши читаются на каждом токене, и они первыми получают
    // общую с DSP память; prefill (AR-128) идёт редко и может остаться с копированием.
    std::vector<int> order = ars_;
    std::sort(order.begin(), order.end());
    for (int n : order)
      for (auto &p : meta_["parts"]) Bind(p["graphs"][std::to_string(n)], n);
    LFM2_LOG("буферы: rpcmem %s, отказов memRegister %d", Rpc().Ok() ? "да" : "нет", qnn_->RegisterFailures());
  }

  ~Engine() {
    qnn_->DeregisterAll();  // до освобождения rpcmem
    for (auto &b : bufs_) b.second.Release();
    for (auto &b : state_) b.second.Release();
  }

  int ContextLength() const { return cl_; }
  int NPast() const { return n_past_; }
  int VocabSize() const { return meta_["vocab_size"]; }

  void Reset() {
    n_past_ = 0;
    for (auto &g : groups_) std::fill(g.second.slot_pos.begin(), g.second.slot_pos.end(), -1);
    for (auto &s : state_) s.second.Fill((f16)0);
  }

  std::vector<float> Append(const std::vector<int> &ids) {
    if (ids.size() == 1) return Step(ids, 1);
    size_t first = ids.size() % ar_ ? ids.size() % ar_ : ar_;
    std::vector<float> logits;
    for (size_t i = 0; i < ids.size();) {
      size_t len = i == 0 ? first : std::min<size_t>(ar_, ids.size() - i);
      std::vector<int> chunk(ids.begin() + i, ids.begin() + i + len);
      if (len == 1 || n_past_ > cl_ - ar_) {
        for (int t : chunk) logits = Step({t}, 1);
      } else {
        logits = Step(chunk, ar_);
      }
      i += len;
    }
    return logits;
  }

 private:
  struct Rope {
    int half = 0;
    std::vector<float> cos, sin;
  };
  struct InputJob {
    std::string role, key;
    Buf *buf;
    json spec;
  };
  struct CacheWrite {
    std::string cache;
    Buf *src;
  };
  struct StateWrite {
    std::string state;
    Buf *src;
  };
  struct Plan {  // всё, что нужно для шага AR-n
    std::vector<GraphRt *> graphs;
    std::vector<InputJob> inputs;
    std::vector<CacheWrite> cache_writes;
    std::vector<StateWrite> state_writes;
    Buf *logits = nullptr;
  };

  int Slots(const std::string &group, int n) const {
    const Group &g = groups_.at(group);
    return g.context ? cl_ - n : g.ring;
  }

  Buf *Named(const std::string &key, size_t elems) {
    Buf &b = bufs_[key];
    if (!b.ptr) b.Alloc(elems);
    if (b.size() != elems) Die("размер буфера " + key);
    return &b;
  }

  void Bind(const std::string &gname, int n) {
    GraphRt *g = graphs_.at(gname).get();
    Plan &plan = plans_[n];
    plan.graphs.push_back(g);
    std::string ns = "@" + std::to_string(n);
    // Сначала входы, затем выходы: перенос (carry) читается из буфера предыдущей части-производителя, а
    // выход пишется в новый буфер этой части — вход и выход одного графа не делят память (HTP пишет на месте).
    for (int pass = 0; pass < 2; ++pass) {
      for (auto &t : meta_["graphs"][gname]["tensors"]) {
        std::string name = t["name"], role = t["role"];
        bool is_in = g->in_idx.count(name) > 0;
        if ((pass == 0) != is_in) continue;
        Tensor *tensor = is_in ? &g->in[g->in_idx[name]] : g->out_idx.count(name) ? &g->out[g->out_idx[name]] : nullptr;
        if (!tensor) Die("в графе " + gname + " нет тензора " + name);
        Buf *b = nullptr;
        if (role == "embedding" || role == "valid") {
          b = Named(role + ":" + name + ns, tensor->elems);
          plan.inputs.push_back({role, name, b, t});
        } else if (role == "mask") {
          std::string key = "mask:" + t["cache_group"].get<std::string>() + ":" + std::to_string(t.value("window", 0)) + ns;
          bool fresh = bufs_.find(key) == bufs_.end();
          b = Named(key, tensor->elems);
          if (fresh) plan.inputs.push_back({role, key, b, t});
        } else if (role == "rope_cos" || role == "rope_sin") {
          std::string key = role + ":" + t["table"].get<std::string>() + ns;
          bool fresh = bufs_.find(key) == bufs_.end();
          b = Named(key, tensor->elems);
          if (fresh) plan.inputs.push_back({role, key, b, t});
        } else if (role == "cache_in") {
          b = Named("cache:" + t["cache"].get<std::string>() + ns, tensor->elems);
        } else if (role == "cache_out") {
          b = Named("new:" + name + ns, tensor->elems);
          plan.cache_writes.push_back({t["cache"], b});
        } else if (role == "carry") {
          auto it = carry_[n].find(name);
          if (it == carry_[n].end()) Die("перенос " + name + " графа " + gname + " не выдан предыдущей частью");
          b = &bufs_.at(it->second);
        } else if (role == "carry_out") {
          std::string to = t["to"], key = "carry:" + to + ns + "#" + gname;
          b = Named(key, tensor->elems);
          carry_[n][to] = key;
        } else if (role == "state_in") {
          b = &state_[t["state"]];
          if (b->size() != tensor->elems) Die("размер состояния " + name);
        } else if (role == "state_out") {
          b = Named("stateout:" + name + ns, tensor->elems);
          plan.state_writes.push_back({t["state"], b});
        } else if (role == "logits") {
          b = Named("logits" + ns, tensor->elems);
          plan.logits = b;
        } else {
          Die("неизвестная роль " + role);
        }
        ::Bind(qnn_.get(), g, tensor, b);
      }
    }
    g->in_raw.clear();
    g->out_raw.clear();
    for (auto &t : g->in) {
      if (t.t.v1.memType == QNN_TENSORMEMTYPE_RAW && !t.t.v1.clientBuf.data) Die("вход " + t.name + " графа " + gname + " не описан в metadata");
      g->in_raw.push_back(t.t);
    }
    for (auto &t : g->out) {
      if (t.t.v1.memType == QNN_TENSORMEMTYPE_RAW && !t.t.v1.clientBuf.data) Die("выход " + t.name + " графа " + gname + " не описан в metadata");
      g->out_raw.push_back(t.t);
    }
  }

  std::vector<float> Step(const std::vector<int> &ids, int n) {
    int real = ids.size(), pad = n - real;
    if (n_past_ + real > cl_ || (n > 1 && n_past_ > cl_ - n)) Die("контекст переполнен");
    std::vector<int64_t> pos(n, 0);
    for (int r = pad; r < n; ++r) pos[r] = n_past_ + r - pad;
    Plan &plan = plans_.at(n);
    for (auto &job : plan.inputs) {
      f16 *dst = job.buf->data();
      if (job.role == "embedding") {
        const EmbeddingTable &e = *emb_.at(job.spec["source"].get<std::string>());
        job.buf->Fill((f16)0);
        for (int r = 0; r < real; ++r) e.Row(ids[r], dst + (size_t)(pad + r) * e.dim);
      } else if (job.role == "valid") {
        for (int r = 0; r < n; ++r) dst[r] = (f16)(r < pad ? 0.0f : 1.0f);
      } else if (job.role == "rope_cos" || job.role == "rope_sin") {
        const Rope &rp = rope_.at(job.spec["table"].get<std::string>());
        const std::vector<float> &tab = job.role == "rope_cos" ? rp.cos : rp.sin;
        int h = rp.half;
        for (int r = 0; r < n; ++r)
          for (int d = 0; d < h; ++d) dst[r * 2 * h + d] = dst[r * 2 * h + h + d] = (f16)tab[(size_t)pos[r] * h + d];
      } else if (job.role == "mask") {
        std::string group = job.spec["cache_group"];
        int window = job.spec.value("window", 0);
        int S = Slots(group, n);
        const std::vector<int64_t> &sp = groups_.at(group).slot_pos;
        for (int r = 0; r < n; ++r) {
          f16 *m = dst + (size_t)r * (S + n);
          std::fill(m, m + S + n, (f16)kMaskNeg);
          if (r < pad) {
            m[S + r] = 0;
            continue;
          }
          int64_t q = pos[r];
          for (int j = 0; j < S; ++j)
            if (sp[j] >= 0 && sp[j] <= q && (!window || q - sp[j] < window)) m[j] = 0;
          int lo = window ? std::max(pad, r - window + 1) : pad;
          for (int c = lo; c <= r; ++c) m[S + c] = 0;
        }
      }
    }
    for (GraphRt *g : plan.graphs) qnn_->Execute(g);
    for (auto &w : plan.cache_writes) {
      const CacheSpec &c = caches_.at(w.cache);
      const Group &grp = groups_.at(c.group);
      const f16 *src = w.src->data();
      for (int nn : ars_) {
        int S = Slots(c.group, nn);
        f16 *dst = bufs_.at("cache:" + w.cache + "@" + std::to_string(nn)).data();
        for (int r = pad; r < n; ++r) {
          int64_t slot = grp.context ? pos[r] : pos[r] % grp.ring;
          if (slot >= S) continue;
          for (int h = 0; h < c.heads; ++h) {
            if (c.bhdt) {
              for (int d = 0; d < c.dim; ++d)
                dst[((size_t)h * c.dim + d) * S + slot] = src[((size_t)h * c.dim + d) * n + r];
            } else {
              memcpy(&dst[((size_t)h * S + slot) * c.dim], &src[((size_t)h * n + r) * c.dim], c.dim * sizeof(f16));
            }
          }
        }
      }
    }
    for (auto &g : groups_)
      for (int r = pad; r < n; ++r) {
        int64_t slot = g.second.context ? pos[r] : pos[r] % g.second.ring;
        if (slot < (int64_t)g.second.slot_pos.size()) g.second.slot_pos[slot] = pos[r];
      }
    for (auto &w : plan.state_writes) memcpy(state_[w.state].data(), w.src->data(), w.src->Bytes());
    n_past_ += real;
    std::vector<float> logits(plan.logits->size());
    const f16 *lg = plan.logits->data();
    for (size_t i = 0; i < logits.size(); ++i) logits[i] = (float)lg[i];
    return logits;
  }

  json meta_;
  int cl_ = 0, ar_ = 1, n_past_ = 0;
  std::vector<int> ars_;
  std::unique_ptr<Qnn> qnn_;
  std::map<std::string, std::unique_ptr<GraphRt>> graphs_;
  std::map<std::string, std::unique_ptr<EmbeddingTable>> emb_;
  std::map<std::string, Rope> rope_;
  std::map<std::string, Group> groups_;
  std::map<std::string, CacheSpec> caches_;
  std::map<std::string, Buf> state_;
  std::map<std::string, Buf> bufs_;
  std::map<int, std::map<std::string, std::string>> carry_;  // n -> имя входа переноса -> ключ буфера производителя
  std::map<int, Plan> plans_;
};

}  // namespace

// ---------------------------------------------------------------------------------------------------
// C-API: ошибка — код < 0 или nullptr, текст — QnnLlmLastError / QnnLlmCreateError.
// ---------------------------------------------------------------------------------------------------
struct QnnLlmEngine {
  std::unique_ptr<Engine> engine;
  std::vector<float> logits;
  std::vector<int> history;
  std::mt19937_64 rng{42};
  std::string error;
  std::mutex mu;
};

#define QNN_LLM_API extern "C" __attribute__((visibility("default")))

namespace {
thread_local std::string g_create_error;
}

QNN_LLM_API void QnnLlmPrependAdspLibraryPath(const char *directory) {
  const char *old = getenv("ADSP_LIBRARY_PATH");
  std::string value = directory;
  if (old && *old) value += std::string(";") + old;
  setenv("ADSP_LIBRARY_PATH", value.c_str(), 1);
}

// libDir — каталог libQnnHtp.so/libQnnSystem.so; modelDir — пакет qnn-llm-convert (metadata.json, part*.bin, …).
QNN_LLM_API QnnLlmEngine *QnnLlmCreate(const char *libDir, const char *modelDir) {
  try {
    auto e = std::make_unique<QnnLlmEngine>();
    e->engine = std::make_unique<Engine>(libDir, modelDir);
    return e.release();
  } catch (const std::exception &ex) {
    g_create_error = ex.what();
    LFM2_LOG("create failed: %s", ex.what());
    return nullptr;
  }
}

QNN_LLM_API const char *QnnLlmCreateError() { return g_create_error.c_str(); }
QNN_LLM_API const char *QnnLlmLastError(QnnLlmEngine *e) { return e ? e->error.c_str() : ""; }
QNN_LLM_API int QnnLlmContextLength(QnnLlmEngine *e) { return e->engine->ContextLength(); }
QNN_LLM_API int QnnLlmVocabSize(QnnLlmEngine *e) { return e->engine->VocabSize(); }

QNN_LLM_API int QnnLlmNPast(QnnLlmEngine *e) {
  std::lock_guard<std::mutex> lock(e->mu);
  return e->engine->NPast();
}

QNN_LLM_API void QnnLlmReset(QnnLlmEngine *e) {
  std::lock_guard<std::mutex> lock(e->mu);
  e->engine->Reset();
  e->history.clear();
  e->logits.clear();
}

// Добавляет токены в контекст (prefill или шаг decode). 0 — успех, -1 — не хватает контекста, -2 — ошибка QNN.
QNN_LLM_API int QnnLlmAppend(QnnLlmEngine *e, const int *ids, int count) {
  std::lock_guard<std::mutex> lock(e->mu);
  if (count <= 0) return 0;
  if (e->engine->NPast() + count > e->engine->ContextLength()) {
    e->error = "контекст переполнен";
    return -1;
  }
  try {
    std::vector<int> v(ids, ids + count);
    e->logits = e->engine->Append(v);
    e->history.insert(e->history.end(), v.begin(), v.end());
    return 0;
  } catch (const std::exception &ex) {
    e->error = ex.what();
    LFM2_LOG("append failed: %s", ex.what());
    return -2;
  }
}

// Следующий токен по логитам последнего Append: штраф за повторы (по токенам контекста), температура
// (<= 0 — жадно), min-p относительно лучшего. -1 — логитов нет.
QNN_LLM_API int QnnLlmSample(QnnLlmEngine *e, float temperature, float minP, float repetitionPenalty, uint64_t seed) {
  std::lock_guard<std::mutex> lock(e->mu);
  if (e->logits.empty()) {
    e->error = "нет логитов: сначала QnnLlmAppend";
    return -1;
  }
  std::vector<float> l = e->logits;
  if (repetitionPenalty > 1.0f) {
    std::vector<bool> seen(l.size(), false);
    for (int t : e->history)
      if (t >= 0 && t < (int)l.size() && !seen[t]) {
        seen[t] = true;
        l[t] = l[t] > 0 ? l[t] / repetitionPenalty : l[t] * repetitionPenalty;
      }
  }
  int best = 0;
  for (size_t i = 1; i < l.size(); ++i)
    if (l[i] > l[best]) best = i;
  if (temperature <= 0.0f) return best;
  if (seed) e->rng.seed(seed);
  double mx = l[best] / temperature, total = 0;
  std::vector<double> p(l.size());
  for (size_t i = 0; i < l.size(); ++i) {
    p[i] = std::exp(l[i] / temperature - mx);  // у лучшего p = 1, поэтому порог min-p — просто minP
    if (p[i] < minP) p[i] = 0;
    total += p[i];
  }
  std::uniform_real_distribution<double> u(0.0, total);
  double r = u(e->rng);
  for (size_t i = 0; i < p.size(); ++i) {
    r -= p[i];
    if (r <= 0 && p[i] > 0) return i;
  }
  return best;
}

QNN_LLM_API void QnnLlmDestroy(QnnLlmEngine *e) { delete e; }
