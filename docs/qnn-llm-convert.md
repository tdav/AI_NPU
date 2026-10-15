# qnn-llm-convert — конвертер LLM onnx-community → QNN HTP (Snapdragon NPU)

Консольный инструмент: `qnn-llm-convert --url <HF-репозиторий onnx-community> --quant q4` скачивает выбранный
вариант ONNX, пересобирает его в статические графы для NPU Qualcomm, компилирует context binary и загружает
результат в новый бакет HuggingFace. Рантайм (`native/lfm2-npu` → универсальный) и приложение AiNpu запускают
такие модели по `metadata.json`.

## Состав

| Часть | Где | Что делает |
|---|---|---|
| Ядро | `converter/qnnllm/` (Python 3.10) | скачивание, адаптер архитектуры, статический ONNX + энкодинги INT4, проверка в ORT, qairt-converter, context binary, упаковка, загрузка |
| Адаптеры | `converter/qnnllm/arch/` | `lfm2` (LFM2/LFM2.5), `gemma4_text` (Gemma 4 E2B/E4B, только текст) |
| Обёртка Windows | `converter/win/` (.NET, single-file exe) | ставит Python 3.10 через uv и зависимости, находит QAIRT SDK, запускает ядро |
| Рантайм | `native/lfm2-npu` | исполняет части по `metadata.json` (роли входов: эмбеддинги, маски, RoPE, KV-кэши, состояния) |

Требования: QAIRT SDK 2.40.0.251030 (по лицензии Qualcomm в инструмент не входит), 32 ГБ свободной ОЗУ,
~60 ГБ на диске для модели класса 4B, токен HuggingFace для загрузки (`hf auth login` или `HF_TOKEN`).

## Запуск

Linux / WSL (Python 3.10, `pip install -r converter/requirements.txt`):

```bash
PYTHONPATH=converter python -m qnnllm --url https://huggingface.co/onnx-community/gemma-4-E4B-it-ONNX --quant q4 \
  --qairt ~/qairt/qairt/2.40.0.251030
```

Windows 11 x64 — `artifacts/qnn-llm-convert/qnn-llm-convert.exe` (один файл; при первом запуске сам ставит uv,
Python 3.10 и зависимости в `%LOCALAPPDATA%\qnn-llm-convert`):

```bat
qnn-llm-convert.exe --url https://huggingface.co/onnx-community/LFM2-2.6B-ONNX --quant q4 ^
  --qairt C:\Qualcomm\AIStack\QAIRT\2.40.0.251030 --wsl-qairt ~/qairt/qairt/2.40.0.251030
```

QAIRT 2.40 для Windows не компилирует context binary под мобильные NPU (`HtpPrepare.dll`: «Unsupported HTP
Arch 75» — он рассчитан на Snapdragon X), поэтому последний шаг на Windows выполняется Linux-версией QAIRT в WSL
(`--wsl-qairt`, при необходимости `--wsl-env` — скрипт окружения). Скачивание, сборка частей, сверка и
конвертация в DLC идут на Windows нативно.

Повторный запуск с тем же рабочим каталогом продолжает с незавершённого шага; `--from-step` — начать с нужного.

## Результаты (SM8650, Xiaomi 23117RK66C)

| Модель | Пакет | Сверка с ONNX (ORT) | Prefill | Decode |
|---|---|---|---|---|
| LFM2-2.6B q4 | 1,8 ГБ, 3 части | top-1 25/25, KL 0,007 | ≈500 ток/с | ≈11 ток/с |
| Gemma 4 E4B q4 (текст) | 4,3 ГБ, 5 частей + эмбеддинги | top-1 13/13, KL 0,004 (промпт > 512 токенов) | ≈180 ток/с | ≈5,5 ток/с |

## Почему графы пересобираются, а не конвертируются как есть

Экспорт onnx-community рассчитан на onnxruntime: динамические формы, растущий KV-кэш, операции
`com.microsoft` (MatMulNBits, GroupQueryAttention, RotaryEmbedding, GatherBlockQuantized). HTP требует
статических форм и стандартных операций, поэтому адаптер строит граф заново по весам и семантике исходного ONNX:

- линейные слои — MatMul с весами на сетке знакового INT4 block-32 (энкодинги `PER_BLOCK` через
  `--quantization_overrides`), активации FP16 — без калибровки;
- KV-кэш фиксированного размера: вход `past_*` (P слотов), выход — только новые K/V, хост пишет их в слоты;
- маски и RoPE считает хост и подаёт входами; скользящее окно — кольцевой кэш на W слотов;
- модель режется на части по слоям (≈0,5 ГБ на context binary); в каждой части два графа с общими весами:
  AR-128 (prefill) и AR-1 (decode).

### Квантизация весов

HTP v75 исполняет FullyConnected «FP16 × INT4» только для **симметричных** блоков со знаком (сетка −8…7,
положительный scale). Варианты q4 у onnx-community бывают:

- симметричные (MatMulNBits без zero point, LFM2): блоки с отрицательным scale переворачиваются без потерь;
  блоки, где после переворота появляется +8, переквантуются по минимуму MSE;
- несимметричные (с zero point, Gemma 4): блок с zp = 8 переносится без потерь, остальные переквантуются
  в симметричную сетку по минимуму MSE к весам q4.

Качество проверяется сравнением с исходным ONNX в onnxruntime (совпадение top-1, KL логитов) до компиляции.

## metadata.json (контракт с рантаймом)

```json
{
  "format": "qnn-llm/1",
  "arch": "gemma4_text",
  "source": "onnx-community/gemma-4-E4B-it-ONNX", "quant": "q4",
  "soc": "SM8650", "qairt": "2.40.0.251030",
  "context_length": 4096, "ar": [128, 1],
  "bos": 2, "eos": [1, 106],
  "parts": [{"file": "part0.bin", "graphs": {"128": "p0_ar128", "1": "p0_ar1"}}],
  "embeddings": [{"name": "tok", "file": "embed_tok.q4", "rows": 262144, "dim": 2560, "scale": 50.596}],
  "rope": [{"name": "local", "file": "rope_local.f16", "dim": 256}],
  "caches": [{"name": "k0", "slots": "context", "layout": "BHDT", "shape": [1, 2, 256]}],
  "tensors": [
    {"name": "hidden_in", "role": "embedding", "source": "tok"},
    {"name": "mask_local", "role": "mask", "window": 512, "caches": "ring512"},
    {"name": "past_key_0", "role": "cache_in", "cache": "k0"},
    {"name": "new_key_0", "role": "cache_out", "cache": "k0"},
    {"name": "logits", "role": "logits"}
  ]
}
```

Полная схема — в `converter/qnnllm/metadata.py`.
