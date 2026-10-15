"""README.md пакета модели (карточка бакета HuggingFace)."""
import os


def write_readme(pkg, meta, repo_id):
    v = meta.get("verification")
    files = "\n".join(f"| `{f}` | {d['bytes']:,} | `{d['sha256']}` |".replace(",", " ")
                      for f, d in meta["files"].items())
    parts = "\n".join(f"- `{p['file']}` — слои {p['layers'][0]}–{p['layers'][-1]}, графы "
                      + ", ".join(f"`{g}`" for g in p["graphs"].values()) for p in meta["parts"])
    check = (f"Сверка с исходной моделью в onnxruntime (промпт «{v['prompt']}», {v['steps']} шагов, teacher forcing): "
             f"совпадение top-1 {v['top1']}/{v['steps']}, KL логитов средний {v['mean_kl']:.4f}, максимальный {v['max_kl']:.4f}."
             if v else "Сверка в onnxruntime не выполнялась (--skip-verify).")
    text = f"""---
license: other
base_model: {repo_id}
tags: [qnn, qualcomm, npu, hexagon, android, {meta['arch']}]
pipeline_tag: text-generation
---

# {repo_id.split('/')[1]} ({meta['quant']}) — QNN context binary для NPU {meta['soc']}

Модель [{repo_id}](https://huggingface.co/{repo_id}) (вариант `{meta['quant']}`), пересобранная инструментом
`qnn-llm-convert` (репозиторий AI_NPU) в context binary QNN HTP для {meta['soc']}, QAIRT {meta['qairt']}.
Только текстовая часть. Контекст {meta['context_length']} токенов, графы prefill AR-128 и decode AR-1 с общими весами.

**Изменено относительно оригинала:** граф переписан под статические формы QNN; веса INT4 переупакованы в
симметричные блоки (переквантовано блоков: {meta.get('requantized_blocks', '?')}); эмбеддинги и RoPE
вынесены на хост. Лицензия исходной модели — в файлах LICENSE/README исходного репозитория и действует для этой сборки.

## Части

{parts}

Интерфейс входов/выходов и логика хоста — `metadata.json` (поля `graphs[*].tensors`, роли описаны в
`converter/qnnllm/host.py` репозитория AI_NPU).

## Проверка

{check}

## Файлы

| Файл | Байт | SHA256 |
|---|---|---|
{files}
"""
    with open(os.path.join(pkg, "README.md"), "w") as f:
        f.write(text)
