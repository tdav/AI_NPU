// lfm2-npu-cli: проверка liblfm2-npu на телефоне через adb (жадная генерация по токенам промпта).
// lfm2-npu-cli <каталог libQnn*> <каталог модели> <ids через запятую> [max_new] [temperature]
// Вывод: "IDS: ..." и "STAT: ...". Каталог libQnn* дополнительно ставится в ADSP_LIBRARY_PATH.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

struct Lfm2NpuEngine;
extern "C" {
void Lfm2NpuPrependAdspLibraryPath(const char *directory);
Lfm2NpuEngine *Lfm2NpuCreate(const char *libDir, const char *modelDir, int contextLength, int ar);
const char *Lfm2NpuCreateError();
const char *Lfm2NpuLastError(Lfm2NpuEngine *e);
int Lfm2NpuAppend(Lfm2NpuEngine *e, const int *ids, int count);
int Lfm2NpuSample(Lfm2NpuEngine *e, float temperature, float minP, float repetitionPenalty, uint64_t seed);
void Lfm2NpuDestroy(Lfm2NpuEngine *e);
}

static double Now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s <libdir> <modeldir> <ids> [max_new] [temperature]\n", argv[0]);
    return 2;
  }
  std::vector<int> ids;
  std::stringstream ss(argv[3]);
  for (std::string x; std::getline(ss, x, ',');)
    if (!x.empty()) ids.push_back(std::stoi(x));
  int max_new = argc > 4 ? atoi(argv[4]) : 128;
  float temperature = argc > 5 ? atof(argv[5]) : 0.0f;

  Lfm2NpuPrependAdspLibraryPath(argv[1]);
  double t0 = Now();
  Lfm2NpuEngine *e = Lfm2NpuCreate(argv[1], argv[2], 4096, 128);
  if (!e) {
    fprintf(stderr, "ERROR: %s\n", Lfm2NpuCreateError());
    return 1;
  }
  fprintf(stderr, "init %.2fs\n", Now() - t0);
  t0 = Now();
  if (Lfm2NpuAppend(e, ids.data(), ids.size()) != 0) {
    fprintf(stderr, "ERROR: %s\n", Lfm2NpuLastError(e));
    return 1;
  }
  double tp = Now() - t0;
  std::vector<int> out;
  t0 = Now();
  for (int i = 0; i < max_new; ++i) {
    int t = Lfm2NpuSample(e, temperature, 0.15f, 1.05f, 0);
    if (t < 0 || t == 7) break;
    out.push_back(t);
    if (Lfm2NpuAppend(e, &t, 1) != 0) {
      fprintf(stderr, "ERROR: %s\n", Lfm2NpuLastError(e));
      return 1;
    }
  }
  double td = Now() - t0;
  printf("IDS:");
  for (size_t i = 0; i < out.size(); ++i) printf("%s%d", i ? "," : "", out[i]);
  printf("\nSTAT: prompt=%zu prefill=%.3fs (%.1f tok/s) gen=%zu decode=%.3fs (%.2f tok/s)\n", ids.size(), tp,
         ids.size() / tp, out.size(), td, out.size() > 1 ? out.size() / td : 0.0);
  Lfm2NpuDestroy(e);
  return 0;
}
