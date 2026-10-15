// liblfm2-npu: инференс LFM2-2.6B (q4) на NPU Qualcomm HTP (QNN 2.40, SM8650) из context binary
// lfm2_p{0,1,2}.bin (бакет tdavron/lfm2-2.6b-q4-qnn-sm8650-bucket). Хост держит KV-кэш фиксированного
// размера CL, conv-состояния, аддитивную маску и RoPE; prefill идёт чанками по AR (левый паддинг),
// decode — по одному токену. C-API (Lfm2Npu*) — для P/Invoke из приложения AiNpu.
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "HTP/QnnHtpDevice.h"
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

using f16 = _Float16;

namespace {

constexpr int kHid = 2048, kKv = 8, kHd = 64, kVocab = 65536;
constexpr float kMaskNeg = -10000.0f;

#ifdef __ANDROID__
#include <android/log.h>
#define LFM2_LOG(...) __android_log_print(ANDROID_LOG_INFO, "lfm2-npu", __VA_ARGS__)
#else
#define LFM2_LOG(...) (fprintf(stderr, __VA_ARGS__), fputc('\n', stderr))
#endif

[[noreturn]] void Die(const std::string &msg) { throw std::runtime_error(msg); }

void LogCb(const char *fmt, QnnLog_Level_t level, uint64_t, va_list args) {
  if (level > (getenv("LFM2_QNN_VERBOSE") ? QNN_LOG_LEVEL_VERBOSE : QNN_LOG_LEVEL_WARN)) return;
#ifdef __ANDROID__
  __android_log_vprint(ANDROID_LOG_WARN, "lfm2-npu-qnn", fmt, args);
#else
  vfprintf(stderr, fmt, args);
  fputc('\n', stderr);
#endif
}

double Now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::vector<std::string> Split(const std::string &s, char c) {
  std::vector<std::string> r;
  std::stringstream ss(s);
  std::string x;
  while (std::getline(ss, x, c)) {
    if (!x.empty()) r.push_back(x);
  }
  return r;
}

struct Buf {  // буфер тензора (FP16), может разделяться между графами
  std::vector<f16> data;
  size_t Bytes() const { return data.size() * sizeof(f16); }
};

struct Tensor {
  Qnn_Tensor_t t;
  std::string name;
  std::vector<uint32_t> dims;
  size_t elems = 1;
};

struct GraphRt {
  std::string name;
  Qnn_GraphHandle_t handle = nullptr;
  std::vector<Tensor> in, out;
  std::vector<Qnn_Tensor_t> in_raw, out_raw;
  std::map<std::string, size_t> in_idx, out_idx;
};

class Qnn {
 public:
  Qnn(const std::string &backend, const std::string &system) {
    lib_ = dlopen(backend.c_str(), RTLD_NOW | RTLD_LOCAL);
    void *lib = lib_;
    if (!lib) Die(std::string("dlopen backend: ") + dlerror());
    auto get = reinterpret_cast<decltype(&QnnInterface_getProviders)>(dlsym(lib, "QnnInterface_getProviders"));
    const QnnInterface_t **prov = nullptr;
    uint32_t n = 0;
    if (!get || get(&prov, &n) != QNN_SUCCESS || n == 0) Die("QnnInterface_getProviders");
    qnn_ = prov[0]->QNN_INTERFACE_VER_NAME;
    slib_ = dlopen(system.c_str(), RTLD_NOW | RTLD_LOCAL);
    void *slib = slib_;
    if (!slib) Die(std::string("dlopen system: ") + dlerror());
    auto sget = reinterpret_cast<decltype(&QnnSystemInterface_getProviders)>(
        dlsym(slib, "QnnSystemInterface_getProviders"));
    const QnnSystemInterface_t **sprov = nullptr;
    if (!sget || sget(&sprov, &n) != QNN_SUCCESS || n == 0) Die("QnnSystemInterface_getProviders");
    sys_ = sprov[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
    if (qnn_.logCreate(LogCb, getenv("LFM2_QNN_VERBOSE") ? QNN_LOG_LEVEL_VERBOSE : QNN_LOG_LEVEL_WARN, &log_) != QNN_SUCCESS) log_ = nullptr;
    if (qnn_.backendCreate(log_, nullptr, &backend_) != QNN_SUCCESS) Die("backendCreate");
    if (qnn_.deviceCreate && qnn_.deviceCreate(log_, nullptr, &device_) != QNN_SUCCESS) device_ = nullptr;
    SetBurst();
  }

  ~Qnn() {
    for (auto c : contexts_) qnn_.contextFree(c, nullptr);
    if (device_ && qnn_.deviceFree) qnn_.deviceFree(device_);
    if (backend_) qnn_.backendFree(backend_);
    if (log_) qnn_.logFree(log_);
    // dlclose не делаем: QNN HTP держит потоки, выгрузка библиотеки во время их работы опасна.
  }

  // Голос за максимальные частоты HTP (без него NPU работает на низких частотах DCVS).
  void SetBurst() {
    QnnDevice_Infrastructure_t infra = nullptr;
    if (!qnn_.deviceGetInfrastructure || qnn_.deviceGetInfrastructure(&infra) != QNN_SUCCESS || !infra) {
      LFM2_LOG("perf: нет device infrastructure (эмулятор?)");
      return;
    }
    auto *htp = reinterpret_cast<QnnHtpDevice_Infrastructure_t *>(infra);
    if (htp->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) return;
    QnnHtpDevice_PerfInfrastructure_t perf = htp->perfInfra;
    uint32_t id = 0;
    if (perf.createPowerConfigId(0, 0, &id) != QNN_SUCCESS) {
      LFM2_LOG("perf: createPowerConfigId failed");
      return;
    }
    QnnHtpPerfInfrastructure_PowerConfig_t dcvs{}, lat{}, poll{};
    dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    auto &d = dcvs.dcvsV3Config;
    d.contextId = id;
    d.setDcvsEnable = 1;
    d.dcvsEnable = 0;
    d.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    d.setSleepLatency = 1;
    d.sleepLatency = 40;
    d.setSleepDisable = 1;
    d.sleepDisable = 1;
    d.setBusParams = 1;
    d.busVoltageCornerMin = d.busVoltageCornerTarget = d.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    d.setCoreParams = 1;
    d.coreVoltageCornerMin = d.coreVoltageCornerTarget = d.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    lat.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    lat.rpcControlLatencyConfig = 100;
    poll.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    poll.rpcPollingTimeConfig = 9999;
    const QnnHtpPerfInfrastructure_PowerConfig_t *cfg[] = {&dcvs, &lat, &poll, nullptr};
    Qnn_ErrorHandle_t e = perf.setPowerConfig(id, cfg);
    LFM2_LOG("perf: burst %s", e == QNN_SUCCESS ? "on" : "FAILED");
  }

  // Загружает context binary и все графы в нём.
  std::vector<std::unique_ptr<GraphRt>> Load(const std::string &path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) Die("open " + path);
    struct stat st;
    fstat(fd, &st);
    size_t size = st.st_size;
    void *mem = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mem == MAP_FAILED) Die("mmap " + path);

    QnnSystemContext_Handle_t sctx = nullptr;
    if (sys_.systemContextCreate(&sctx) != QNN_SUCCESS) Die("systemContextCreate");
    const QnnSystemContext_BinaryInfo_t *info = nullptr;
    Qnn_ContextBinarySize_t info_size = 0;
    if (sys_.systemContextGetBinaryInfo(sctx, mem, size, &info, &info_size) != QNN_SUCCESS)
      Die("systemContextGetBinaryInfo " + path);
    // V1/V2/V3 совпадают в начале структуры (numGraphs/graphs), как и GraphInfo V1/V2/V3.
    uint32_t ngraphs = info->contextBinaryInfoV1.numGraphs;
    QnnSystemContext_GraphInfo_t *ginfo = info->contextBinaryInfoV1.graphs;
    if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
      ngraphs = info->contextBinaryInfoV2.numGraphs;
      ginfo = info->contextBinaryInfoV2.graphs;
    } else if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
      ngraphs = info->contextBinaryInfoV3.numGraphs;
      ginfo = info->contextBinaryInfoV3.graphs;
    }
    std::vector<std::unique_ptr<GraphRt>> graphs;
    for (uint32_t i = 0; i < ngraphs; ++i) {
      const auto &g = ginfo[i].graphInfoV1;
      auto rt = std::make_unique<GraphRt>();
      rt->name = g.graphName;
      CopyTensors(g.graphInputs, g.numGraphInputs, &rt->in, &rt->in_idx);
      CopyTensors(g.graphOutputs, g.numGraphOutputs, &rt->out, &rt->out_idx);
      graphs.push_back(std::move(rt));
    }
    sys_.systemContextFree(sctx);

    Qnn_ContextHandle_t ctx = nullptr;
    double t0 = Now();
    if (qnn_.contextCreateFromBinary(backend_, device_, nullptr, mem, size, &ctx, nullptr) != QNN_SUCCESS)
      Die("contextCreateFromBinary " + path);
    munmap(mem, size);
    for (auto &g : graphs) {
      if (qnn_.graphRetrieve(ctx, g->name.c_str(), &g->handle) != QNN_SUCCESS) Die("graphRetrieve " + g->name);
    }
    contexts_.push_back(ctx);
    LFM2_LOG("loaded %s (%zu MB, %d graphs) in %.2fs", path.c_str(), size >> 20, (int)graphs.size(), Now() - t0);
    return graphs;
  }

  void Execute(GraphRt *g) {
    if (qnn_.graphExecute(g->handle, g->in_raw.data(), g->in_raw.size(), g->out_raw.data(), g->out_raw.size(),
                          nullptr, nullptr) != QNN_SUCCESS)
      Die("graphExecute " + g->name);
  }

 private:
  static void CopyTensors(const Qnn_Tensor_t *src, uint32_t n, std::vector<Tensor> *dst,
                          std::map<std::string, size_t> *idx) {
    dst->resize(n);
    for (uint32_t i = 0; i < n; ++i) {
      Tensor &t = (*dst)[i];
      t.t = src[i];
      const Qnn_TensorV1_t &v = src[i].v1;  // V2 начинается с полей V1
      t.name = v.name;
      t.dims.assign(v.dimensions, v.dimensions + v.rank);
      for (auto d : t.dims) t.elems *= d;
      if (v.dataType != QNN_DATATYPE_FLOAT_16) Die("tensor " + t.name + ": ожидается FP16");
      (*idx)[t.name] = i;
    }
    for (auto &t : *dst) {  // указатели на собственные копии
      t.t.v1.name = t.name.c_str();
      t.t.v1.dimensions = t.dims.data();
      t.t.v1.memType = QNN_TENSORMEMTYPE_RAW;
      if (t.t.version == QNN_TENSOR_VERSION_2) t.t.v2.isDynamicDimensions = nullptr;
    }
  }

  QNN_INTERFACE_VER_TYPE qnn_{};
  QNN_SYSTEM_INTERFACE_VER_TYPE sys_{};
  Qnn_LogHandle_t log_ = nullptr;
  Qnn_BackendHandle_t backend_ = nullptr;
  Qnn_DeviceHandle_t device_ = nullptr;
  void *lib_ = nullptr, *slib_ = nullptr;
  std::vector<Qnn_ContextHandle_t> contexts_;
};

void Bind(Tensor *t, Buf *b) {
  if (b->data.size() != t->elems) Die("размер буфера не совпадает для " + t->name);
  t->t.v1.clientBuf.data = b->data.data();
  t->t.v1.clientBuf.dataSize = b->Bytes();
}

class Lfm2 {
 public:
  Lfm2(Qnn *qnn, const std::vector<std::string> &ctx_files, const std::string &embed, int cl, int ar)
      : qnn_(qnn), cl_(cl), ar_(ar) {
    // Графы lfm2_p<часть>_ar<N> могут лежать в одном файле на часть (штатно, с общими весами)
    // или в отдельных файлах (например, для x86-эмулятора, где shared weights не поддерживаются).
    std::map<int, Part> by_part;
    for (auto &f : ctx_files) {
      for (auto &g : qnn->Load(f)) {
        int p = std::stoi(g->name.substr(g->name.find("_p") + 2));
        int n = std::stoi(g->name.substr(g->name.rfind("_ar") + 3));
        by_part[p].g[n] = std::move(g);
      }
    }
    for (auto &it : by_part) {
      if ((int)parts_.size() != it.first) Die("пропущена часть " + std::to_string(parts_.size()));
      if (!it.second.g.count(1) || !it.second.g.count(ar))
        Die("для части " + std::to_string(it.first) + " нет графов ar1/ar" + std::to_string(ar));
      parts_.push_back(std::move(it.second));
    }
    // Эмбеддинги: mmap FP16 [vocab, hid].
    int fd = open(embed.c_str(), O_RDONLY);
    if (fd < 0) Die("open " + embed);
    embed_ = static_cast<const f16 *>(mmap(nullptr, (size_t)kVocab * kHid * 2, PROT_READ, MAP_SHARED, fd, 0));
    close(fd);
    if (embed_ == MAP_FAILED) Die("mmap embed");
    // RoPE
    cos_.resize((size_t)cl * kHd);
    sin_.resize((size_t)cl * kHd);
    for (int p = 0; p < cl; ++p)
      for (int i = 0; i < kHd / 2; ++i) {
        double f = p / std::pow(1e6, 2.0 * i / kHd);
        cos_[p * kHd + i] = cos_[p * kHd + i + kHd / 2] = (float)std::cos(f);
        sin_[p * kHd + i] = sin_[p * kHd + i + kHd / 2] = (float)std::sin(f);
      }
    for (int n : {1, ar}) SetupBuffers(n);
  }

  std::vector<float> Prefill(const std::vector<int> &ids) {
    size_t first = ids.size() % ar_ ? ids.size() % ar_ : ar_;
    std::vector<float> logits;
    for (size_t s = 0; s < ids.size();) {
      size_t len = s == 0 ? first : std::min<size_t>(ar_, ids.size() - s);
      std::vector<int> chunk(ids.begin() + s, ids.begin() + s + len);
      if (len == 1 || n_past_ > cl_ - ar_) {
        for (int id : chunk) logits = Step({id}, 1);  // конец контекста: в кэш AR-N уже не помещается
      } else {
        logits = Step(chunk, ar_);
      }
      s += len;
    }
    return logits;
  }

  std::vector<float> Decode(int id) { return Step({id}, 1); }
  int NPast() const { return n_past_; }
  int ContextLength() const { return cl_; }
  int Ar() const { return ar_; }

  ~Lfm2() {
    if (embed_) munmap(const_cast<f16 *>(embed_), (size_t)kVocab * kHid * 2);
  }

  // Новый диалог: conv-состояния обнуляются, KV-слоты считаются пустыми (их скрывает маска).
  void Reset() {
    n_past_ = 0;
    for (auto &it : io_)
      for (auto &cs : it.second.conv_in) std::fill(cs.second.data.begin(), cs.second.data.end(), (f16)0);
  }

 private:
  struct Part {
    std::map<int, std::unique_ptr<GraphRt>> g;
  };
  struct Io {  // буферы для графов шага N
    Buf hidden, mask, cos, sin, valid, logits;
    std::vector<Buf> hidden_mid;                 // выходы частей 0..P-2
    std::map<int, Buf> past_k, past_v;           // кэш в раскладке графа [1,8,64,P] / [1,8,P,64]
    std::map<int, Buf> new_k, new_v, conv_in, conv_out;
  };

  Buf *Get(std::map<int, Buf> &m, int li, size_t elems) {
    auto &b = m[li];
    if (b.data.empty()) b.data.assign(elems, (f16)0);
    return &b;
  }

  void SetupBuffers(int n) {
    Io &io = io_[n];
    int P = cl_ - n;
    io.hidden.data.assign((size_t)n * kHid, 0);
    io.mask.data.assign((size_t)n * cl_, 0);
    io.cos.data.assign((size_t)n * kHd, 0);
    io.sin.data.assign((size_t)n * kHd, 0);
    io.valid.data.assign(n, 1);
    io.logits.data.assign(kVocab, 0);
    io.hidden_mid.resize(parts_.size());
    for (size_t pi = 0; pi < parts_.size(); ++pi) {
      GraphRt *g = parts_[pi].g[n].get();
      for (auto &t : g->in) {
        const std::string &nm = t.name;
        if (nm == "hidden_in") Bind(&t, pi == 0 ? &io.hidden : &io.hidden_mid[pi - 1]);
        else if (nm == "attn_mask") Bind(&t, &io.mask);
        else if (nm == "rope_cos") Bind(&t, &io.cos);
        else if (nm == "rope_sin") Bind(&t, &io.sin);
        else if (nm == "valid") Bind(&t, &io.valid);
        else if (nm.rfind("past_key_", 0) == 0) Bind(&t, Get(io.past_k, std::stoi(nm.substr(9)), (size_t)kKv * kHd * P));
        else if (nm.rfind("past_value_", 0) == 0) Bind(&t, Get(io.past_v, std::stoi(nm.substr(11)), (size_t)kKv * kHd * P));
        else if (nm.rfind("conv_state_", 0) == 0) Bind(&t, Get(io.conv_in, std::stoi(nm.substr(11)), 2 * kHid));
        else Die("неизвестный вход " + nm);
      }
      for (auto &t : g->out) {
        const std::string &nm = t.name;
        if (nm == "hidden_out") {
          io.hidden_mid[pi].data.assign((size_t)n * kHid, 0);
          Bind(&t, &io.hidden_mid[pi]);
        } else if (nm == "logits") Bind(&t, &io.logits);
        else if (nm.rfind("new_key_", 0) == 0) Bind(&t, Get(io.new_k, std::stoi(nm.substr(8)), (size_t)kKv * kHd * n));
        else if (nm.rfind("new_value_", 0) == 0) Bind(&t, Get(io.new_v, std::stoi(nm.substr(10)), (size_t)kKv * kHd * n));
        else if (nm.rfind("new_conv_state_", 0) == 0) Bind(&t, Get(io.conv_out, std::stoi(nm.substr(15)), 2 * kHid));
        else Die("неизвестный выход " + nm);
      }
      g->in_raw.clear();
      g->out_raw.clear();
      for (auto &t : g->in) g->in_raw.push_back(t.t);
      for (auto &t : g->out) g->out_raw.push_back(t.t);
    }
  }

  std::vector<float> Step(const std::vector<int> &ids, int n) {
    Io &io = io_[n];
    int real = ids.size(), pad = n - real, P = cl_ - n;
    if (n_past_ + real > cl_ || (n > 1 && n_past_ > P)) Die("контекст переполнен");
    std::fill(io.hidden.data.begin(), io.hidden.data.end(), (f16)0);
    for (int r = 0; r < real; ++r)
      memcpy(&io.hidden.data[(size_t)(pad + r) * kHid], embed_ + (size_t)ids[r] * kHid, kHid * 2);
    for (int r = 0; r < n; ++r) {
      int pos = r < pad ? 0 : n_past_ + r - pad;
      for (int d = 0; d < kHd; ++d) {
        io.cos.data[r * kHd + d] = (f16)cos_[pos * kHd + d];
        io.sin.data[r * kHd + d] = (f16)sin_[pos * kHd + d];
      }
      f16 *m = &io.mask.data[(size_t)r * cl_];
      for (int c = 0; c < cl_; ++c) m[c] = (f16)kMaskNeg;
      for (int c = 0; c < n_past_; ++c) m[c] = 0;
      if (r < pad) m[P + r] = 0;
      else for (int c = P + pad; c <= P + r; ++c) m[c] = 0;
      io.valid.data[r] = (f16)(r < pad ? 0.f : 1.f);
    }
    for (size_t pi = 0; pi < parts_.size(); ++pi) qnn_->Execute(parts_[pi].g[n].get());
    // Новые KV -> кэши обоих графов (ar1 и arN), conv-состояния -> входы обоих графов.
    for (auto &kv : io.new_k) {
      int li = kv.first;
      for (auto &it : io_) {
        int Pn = cl_ - it.first;
        Io &dst = it.second;
        f16 *K = dst.past_k[li].data.data(), *V = dst.past_v[li].data.data();
        const f16 *nk = io.new_k[li].data.data(), *nv = io.new_v[li].data.data();
        for (int r = pad; r < n; ++r) {
          int slot = n_past_ + r - pad;
          if (slot >= Pn) continue;  // в кэш этого графа не помещается (не будет использован)
          for (int h = 0; h < kKv; ++h) {
            for (int d = 0; d < kHd; ++d) K[((size_t)h * kHd + d) * Pn + slot] = nk[((size_t)h * kHd + d) * n + r];
            memcpy(&V[((size_t)h * Pn + slot) * kHd], &nv[((size_t)h * n + r) * kHd], kHd * 2);
          }
        }
      }
    }
    for (auto &cs : io.conv_out)
      for (auto &it : io_) memcpy(it.second.conv_in[cs.first].data.data(), cs.second.data.data(), cs.second.Bytes());
    n_past_ += real;
    std::vector<float> logits(kVocab);
    for (int i = 0; i < kVocab; ++i) logits[i] = (float)io.logits.data[i];
    return logits;
  }

  Qnn *qnn_;
  int cl_, ar_, n_past_ = 0;
  std::vector<Part> parts_;
  std::map<int, Io> io_;
  const f16 *embed_ = nullptr;
  std::vector<float> cos_, sin_;
};

}  // namespace

// ---------------------------------------------------------------------------------------------------
// C-API. Все функции потокобезопасны относительно одного движка (внутренний mutex), но генерация
// одного диалога должна идти последовательно. Ошибка: код < 0, текст — Lfm2NpuLastError.
// ---------------------------------------------------------------------------------------------------
struct Lfm2NpuEngine {
  std::unique_ptr<Qnn> qnn;
  std::unique_ptr<Lfm2> model;
  std::vector<float> logits;     // логиты после последнего Append
  std::vector<int> history;      // токены в контексте (для штрафа за повторы)
  std::mt19937_64 rng{42};
  std::string error;
  std::mutex mu;
};

#define LFM2_API extern "C" __attribute__((visibility("default")))

namespace {
thread_local std::string g_create_error;
}

// ADSP_LIBRARY_PATH: разделитель «;», новый путь — первым (DSP ищет skel в каталоге нативных библиотек APK).
LFM2_API void Lfm2NpuPrependAdspLibraryPath(const char *directory) {
  const char *old = getenv("ADSP_LIBRARY_PATH");
  std::string value = directory;
  if (old && *old) value += std::string(";") + old;
  setenv("ADSP_LIBRARY_PATH", value.c_str(), 1);
}

// libDir — каталог с libQnnHtp.so/libQnnSystem.so; modelDir — lfm2_p0..2.bin и embed_tokens.f16.
LFM2_API Lfm2NpuEngine *Lfm2NpuCreate(const char *libDir, const char *modelDir, int contextLength, int ar) {
  try {
    auto e = std::make_unique<Lfm2NpuEngine>();
    std::string lib = libDir, dir = modelDir;
    e->qnn = std::make_unique<Qnn>(lib + "/libQnnHtp.so", lib + "/libQnnSystem.so");
    e->model = std::make_unique<Lfm2>(
        e->qnn.get(), std::vector<std::string>{dir + "/lfm2_p0.bin", dir + "/lfm2_p1.bin", dir + "/lfm2_p2.bin"},
        dir + "/embed_tokens.f16", contextLength, ar);
    return e.release();
  } catch (const std::exception &ex) {
    g_create_error = ex.what();
    LFM2_LOG("create failed: %s", ex.what());
    return nullptr;
  }
}

LFM2_API const char *Lfm2NpuCreateError() { return g_create_error.c_str(); }

LFM2_API const char *Lfm2NpuLastError(Lfm2NpuEngine *e) { return e ? e->error.c_str() : ""; }

LFM2_API int Lfm2NpuContextLength(Lfm2NpuEngine *e) { return e->model->ContextLength(); }

LFM2_API int Lfm2NpuNPast(Lfm2NpuEngine *e) {
  std::lock_guard<std::mutex> lock(e->mu);
  return e->model->NPast();
}

LFM2_API void Lfm2NpuReset(Lfm2NpuEngine *e) {
  std::lock_guard<std::mutex> lock(e->mu);
  e->model->Reset();
  e->history.clear();
  e->logits.clear();
}

// Добавляет токены в контекст (prefill или один шаг decode). 0 — успех, -1 — не хватает контекста, -2 — ошибка QNN.
LFM2_API int Lfm2NpuAppend(Lfm2NpuEngine *e, const int *ids, int count) {
  std::lock_guard<std::mutex> lock(e->mu);
  if (count <= 0) return 0;
  if (e->model->NPast() + count > e->model->ContextLength()) {
    e->error = "контекст переполнен";
    return -1;
  }
  try {
    std::vector<int> v(ids, ids + count);
    e->logits = count == 1 ? e->model->Decode(v[0]) : e->model->Prefill(v);
    e->history.insert(e->history.end(), v.begin(), v.end());
    return 0;
  } catch (const std::exception &ex) {
    e->error = ex.what();
    LFM2_LOG("append failed: %s", ex.what());
    return -2;
  }
}

// Выбор следующего токена по логитам последнего Append: штраф за повторы (как в HF, по всем токенам
// контекста), температура (<= 0 — жадно), min-p. Возвращает id токена или -1, если логитов нет.
LFM2_API int Lfm2NpuSample(Lfm2NpuEngine *e, float temperature, float minP, float repetitionPenalty, uint64_t seed) {
  std::lock_guard<std::mutex> lock(e->mu);
  if (e->logits.empty()) {
    e->error = "нет логитов: сначала Lfm2NpuAppend";
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
  // Ненормированные вероятности: у лучшего токена p = 1, поэтому порог min-p — просто minP.
  double mx = l[best] / temperature;
  std::vector<double> p(l.size());
  for (size_t i = 0; i < l.size(); ++i) p[i] = std::exp(l[i] / temperature - mx);
  double cut = minP > 0 ? minP : 0;
  double total = 0;
  for (auto &x : p) {
    if (x < cut) x = 0;
    total += x;
  }
  std::uniform_real_distribution<double> u(0.0, total);
  double r = u(e->rng);
  for (size_t i = 0; i < p.size(); ++i) {
    r -= p[i];
    if (r <= 0 && p[i] > 0) return i;
  }
  return best;
}

LFM2_API void Lfm2NpuDestroy(Lfm2NpuEngine *e) { delete e; }
