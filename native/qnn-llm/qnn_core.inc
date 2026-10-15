// Общая часть рантайма: загрузка context binary QNN, тензоры графов, голос HTP burst (из lfm2_npu.cc).
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
#define LFM2_LOG(...) __android_log_print(ANDROID_LOG_INFO, "qnn-llm", __VA_ARGS__)
#else
#define LFM2_LOG(...) (fprintf(stderr, __VA_ARGS__), fputc('\n', stderr))
#endif

[[noreturn]] void Die(const std::string &msg) { throw std::runtime_error(msg); }

void LogCb(const char *fmt, QnnLog_Level_t level, uint64_t, va_list args) {
  if (level > (getenv("LFM2_QNN_VERBOSE") ? QNN_LOG_LEVEL_VERBOSE : QNN_LOG_LEVEL_WARN)) return;
#ifdef __ANDROID__
  __android_log_vprint(ANDROID_LOG_WARN, "qnn-llm-qnn", fmt, args);
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

