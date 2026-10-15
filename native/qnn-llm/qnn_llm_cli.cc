// qnn-llm-cli: проверка libqnn-llm на телефоне через adb (жадная генерация по токенам промпта).
// qnn-llm-cli <каталог libQnn*> <каталог модели> <ids через запятую> [max_new] [temperature] [eos,…]
// Вывод: "IDS: ..." и "STAT: ...". Каталог libQnn* дополнительно ставится в ADSP_LIBRARY_PATH.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <sstream>
#include <string>
#include <vector>

struct QnnLlmEngine;
extern "C" {
void QnnLlmPrependAdspLibraryPath(const char *directory);
QnnLlmEngine *QnnLlmCreate(const char *libDir, const char *modelDir);
const char *QnnLlmCreateError();
const char *QnnLlmLastError(QnnLlmEngine *e);
int QnnLlmAppend(QnnLlmEngine *e, const int *ids, int count);
int QnnLlmSample(QnnLlmEngine *e, float temperature, float minP, float repetitionPenalty, uint64_t seed);
void QnnLlmDestroy(QnnLlmEngine *e);
}

static double Now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s <libdir> <modeldir> <ids> [max_new] [temperature] [eos,…]\n", argv[0]);
    return 2;
  }
  std::vector<int> ids;
  std::stringstream ss(argv[3]);
  for (std::string x; std::getline(ss, x, ',');)
    if (!x.empty()) ids.push_back(std::stoi(x));
  int max_new = argc > 4 ? atoi(argv[4]) : 128;
  float temperature = argc > 5 ? atof(argv[5]) : 0.0f;
  std::set<int> eos;  // токены остановки: 6-й аргумент через запятую (по умолчанию 1,7,106 — Gemma и LFM2)
  std::stringstream es(argc > 6 ? argv[6] : "1,7,106");
  for (std::string x; std::getline(es, x, ',');)
    if (!x.empty()) eos.insert(std::stoi(x));

  QnnLlmPrependAdspLibraryPath(argv[1]);
  double t0 = Now();
  QnnLlmEngine *e = QnnLlmCreate(argv[1], argv[2]);
  if (!e) {
    fprintf(stderr, "ERROR: %s\n", QnnLlmCreateError());
    return 1;
  }
  fprintf(stderr, "init %.2fs\n", Now() - t0);
  t0 = Now();
  if (QnnLlmAppend(e, ids.data(), ids.size()) != 0) {
    fprintf(stderr, "ERROR: %s\n", QnnLlmLastError(e));
    return 1;
  }
  double tp = Now() - t0;
  std::vector<int> out;
  t0 = Now();
  for (int i = 0; i < max_new; ++i) {
    int t = QnnLlmSample(e, temperature, 0.15f, 1.05f, 0);
    if (t < 0 || eos.count(t)) break;
    out.push_back(t);
    if (QnnLlmAppend(e, &t, 1) != 0) {
      fprintf(stderr, "ERROR: %s\n", QnnLlmLastError(e));
      return 1;
    }
  }
  double td = Now() - t0;
  printf("IDS:");
  for (size_t i = 0; i < out.size(); ++i) printf("%s%d", i ? "," : "", out[i]);
  printf("\nSTAT: prompt=%zu prefill=%.3fs (%.1f tok/s) gen=%zu decode=%.3fs (%.2f tok/s)\n", ids.size(), tp,
         ids.size() / tp, out.size(), td, out.size() > 1 ? out.size() / td : 0.0);
  QnnLlmDestroy(e);
  return 0;
}
