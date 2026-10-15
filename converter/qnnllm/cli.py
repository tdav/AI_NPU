"""qnn-llm-convert: onnx-community LLM -> QNN HTP context binary -> бакет HuggingFace.

  qnn-llm-convert --url https://huggingface.co/onnx-community/gemma-4-E4B-it-ONNX --quant q4

Шаги (повторный запуск с тем же --work продолжает с незавершённого шага):
  1. download  — config.json, вариант <quant> декодера и эмбеддингов, токенизатор;
  2. build     — статические ONNX-части (AR-128/AR-1) с весами на сетке INT4 + metadata.json;
  3. verify    — сверка с исходной моделью в onnxruntime (top-1, KL логитов);
  4. compile   — qairt-converter -> DLC, qnn-context-binary-generator -> part<i>.bin;
  5. package   — бинарники, эмбеддинги, RoPE, токенизатор, README, LICENSE, SHA256;
  6. upload    — новый бакет <пользователь>/<модель>-<quant>-qnn-<soc>-bucket.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

from . import hub
from .arch import adapter_for
from .build import build_graphs, compare, write_meta
from .qairt import SOCS, Qairt

STEPS = ["download", "build", "verify", "compile", "package", "upload"]

DEFAULT_PROMPT = "Объясни в двух предложениях, что такое нейропроцессор (NPU) в смартфоне."


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def render_prompt(repo_dir, text):
    """Промпт по chat_template модели (jinja) -> ids через tokenizer.json."""
    import jinja2
    from tokenizers import Tokenizer
    tpl_path = os.path.join(repo_dir, "chat_template.jinja")
    tc = json.load(open(os.path.join(repo_dir, "tokenizer_config.json")))
    tpl = open(tpl_path).read() if os.path.exists(tpl_path) else tc["chat_template"]
    tok = lambda k: (tc.get(k) or {}).get("content") if isinstance(tc.get(k), dict) else tc.get(k)
    s = jinja2.Environment().from_string(tpl).render(messages=[{"role": "user", "content": text}],
                                                     add_generation_prompt=True, bos_token=tok("bos_token") or "",
                                                     eos_token=tok("eos_token") or "")
    t = Tokenizer.from_file(os.path.join(repo_dir, "tokenizer.json"))
    return t.encode(s, add_special_tokens=False).ids, t


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qnn-llm-convert", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="репозиторий onnx-community (URL или org/name)")
    ap.add_argument("--quant", default="q4", help="вариант весов: q4 (по умолчанию)")
    ap.add_argument("--soc", default="SM8650", choices=sorted(SOCS), help="целевой SoC (по умолчанию SM8650)")
    ap.add_argument("--context", type=int, default=4096, help="длина контекста, токенов")
    ap.add_argument("--qairt", help="путь к QAIRT SDK (иначе QNN_SDK_ROOT)")
    ap.add_argument("--wsl-qairt", help="Windows: Linux-версия QAIRT в WSL для сборки context binary (QAIRT для Windows не умеет мобильные NPU)")
    ap.add_argument("--wsl-env", help="Windows: скрипт окружения QAIRT в WSL (source перед запуском), если нужен")
    ap.add_argument("--work", help="рабочий каталог (по умолчанию ./work/<модель>-<quant>)")
    ap.add_argument("--bucket", help="имя бакета (по умолчанию <модель>-<quant>-qnn-<soc>-bucket)")
    ap.add_argument("--private", action="store_true", help="приватный бакет")
    ap.add_argument("--no-upload", action="store_true", help="не загружать в HuggingFace")
    ap.add_argument("--skip-verify", action="store_true", help="пропустить сверку в onnxruntime (долго и нужно ~32 ГБ ОЗУ)")
    ap.add_argument("--verify-steps", type=int, default=12, help="шагов генерации при сверке")
    ap.add_argument("--from-step", choices=STEPS, help="начать с шага (по умолчанию — первый незавершённый)")
    a = ap.parse_args(argv)

    repo_id = hub.repo_id_from_url(a.url)
    model = repo_id.split("/")[1].replace("-ONNX", "").replace("-onnx", "")
    work = os.path.abspath(a.work or os.path.join("work", f"{model}-{a.quant}"))
    src, graphs, dlc, pkg = (os.path.join(work, d) for d in ("source", "graphs", "dlc", "package"))
    for d in (src, graphs, dlc, pkg):
        os.makedirs(d, exist_ok=True)
    state_path = os.path.join(work, "state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {"done": []}
    start = STEPS.index(a.from_step) if a.from_step else 0

    def done(step):
        return STEPS.index(step) < start or (not a.from_step and step in state["done"])

    def mark(step, **info):
        state["done"] = sorted(set(state["done"]) | {step}, key=STEPS.index)
        state.update(info)
        json.dump(state, open(state_path, "w"), ensure_ascii=False, indent=1)

    qairt = Qairt(a.qairt, a.wsl_qairt, a.wsl_env)  # проверка SDK до долгих шагов
    log(f"модель {repo_id} ({a.quant}) -> {a.soc}, контекст {a.context}; QAIRT {qairt.version}; каталог {work}")

    hub.fetch_config(repo_id, src)
    cfg = json.load(open(os.path.join(src, "config.json")))
    model_type = cfg.get("text_config", {}).get("model_type") or cfg["model_type"]
    Adapter = adapter_for(model_type)

    if not done("download"):
        log("1/6 скачивание")
        hub.download(repo_id, Adapter.download_patterns(a.quant), src)
        mark("download")
    adapter = Adapter(src, a.quant, a.context)
    meta_path = os.path.join(pkg, "metadata.json")

    if not done("build"):
        log(f"2/6 статические части: {len(adapter.parts)} частей, слои {[(p[0], p[-1]) for p in adapter.parts]}")
        meta = {"format": "qnn-llm/1", "arch": adapter.name, "source": repo_id, "quant": a.quant, "soc": a.soc,
                "qairt": qairt.version, "context_length": a.context, "ar": [128, 1], "vocab_size": adapter.vocab}
        meta.update(adapter.special_tokens())
        meta["chat"] = adapter.chat_format()
        meta.update(build_graphs(adapter, graphs, "npu", log=log))
        meta.update(adapter.host_assets(pkg))
        r = meta.pop("requant")
        meta["requantized_blocks"] = f"{r['requantized']}/{r['blocks']}"
        write_meta(meta_path, meta)
        log(f"  переквантовано блоков INT4: {meta['requantized_blocks']}")
        mark("build")
    meta = json.load(open(meta_path))

    if not done("verify"):
        if a.skip_verify:
            log("3/6 сверка пропущена (--skip-verify)")
        else:
            log("3/6 сверка с исходной моделью в onnxruntime")
            # длинный промпт (>512 токенов): проверяет и prefill чанками, и кольцевые кэши скользящего окна
            ids, tok = render_prompt(src, " ".join([DEFAULT_PROMPT] * 30))
            res = compare(adapter, meta, pkg, graphs, ids, a.verify_steps, log=log)
            log(f"  top-1 {res['top1']}/{res['steps']}, KL средн. {res['mean_kl']:.4f}, макс. {res['max_kl']:.4f}")
            log(f"  эталон: {tok.decode(res['ref_tokens'])!r}")
            meta["verification"] = {k: res[k] for k in ("steps", "top1", "mean_kl", "max_kl")}
            meta["verification"]["prompt"] = f"{DEFAULT_PROMPT} ×30 ({len(ids)} токенов)"
            write_meta(meta_path, meta)
            if res["top1"] < 0.75 * res["steps"]:
                raise SystemExit("Сверка не прошла: top-1 < 75 %. Компиляция остановлена.")
        mark("verify")

    if not done("compile"):
        log("4/6 компиляция QAIRT")
        for pi, part in enumerate(meta["parts"]):
            names = [part["graphs"][str(n)] for n in meta["ar"]]
            paths = []
            for gname in names:
                out = os.path.join(dlc, gname + ".dlc")
                if not os.path.exists(out):
                    t0 = time.time()
                    qairt.convert(os.path.join(graphs, gname + ".onnx"), os.path.join(graphs, gname + ".encodings"), out)
                    log(f"  {gname}.dlc: {os.path.getsize(out) >> 20} МБ, {time.time() - t0:.0f} с")
                paths.append(out)
            t0 = time.time()
            binary = qairt.context_binary(paths, names, a.soc, dlc, f"part{pi}")
            shutil.move(binary, os.path.join(pkg, f"part{pi}.bin"))
            part["file"] = f"part{pi}.bin"
            log(f"  part{pi}.bin: {os.path.getsize(os.path.join(pkg, part['file'])) >> 20} МБ, {time.time() - t0:.0f} с")
        write_meta(meta_path, meta)
        mark("compile")

    if not done("package"):
        log("5/6 упаковка")
        for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                  "config.json", "generation_config.json"):
            if os.path.exists(os.path.join(src, f)):
                shutil.copy(os.path.join(src, f), pkg)
        for f in os.listdir(src):
            if f.upper().startswith("LICENSE"):
                shutil.copy(os.path.join(src, f), pkg)
        files = {f: {"bytes": os.path.getsize(os.path.join(pkg, f)), "sha256": sha256(os.path.join(pkg, f))}
                 for f in sorted(os.listdir(pkg)) if f not in ("metadata.json", "README.md")}
        meta["files"] = files
        write_meta(meta_path, meta)
        from .readme import write_readme
        write_readme(pkg, meta, repo_id)
        mark("package")

    if not done("upload"):
        if a.no_upload:
            log(f"6/6 загрузка пропущена (--no-upload); пакет: {pkg}")
        else:
            name = a.bucket or f"{model.lower()}-{a.quant}-qnn-{a.soc.lower()}-bucket"
            log(f"6/6 загрузка в бакет {name}")
            url = hub.publish(pkg, name, a.private)
            log(f"  готово: {url}")
            mark("upload", bucket_url=url)
    log("готово")


if __name__ == "__main__":
    sys.exit(main())
