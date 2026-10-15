"""Сборка статических ONNX-частей и metadata.json; сверка с исходной моделью в onnxruntime."""
import gc
import json
import os
import time

import numpy as np

from .host import Host


def build_graphs(adapter, outdir, mode, ar=(128, 1), log=print):
    """Строит все части AR-N, сохраняет ONNX (+ .encodings в режиме npu) -> metadata["graphs"/"parts"]."""
    os.makedirs(outdir, exist_ok=True)
    graphs, parts, requant = {}, [], {"blocks": 0, "requantized": 0}
    for pi in range(len(adapter.parts)):
        entry = {"graphs": {}, "layers": [int(i) for i in adapter.parts[pi]]}
        for n in ar:
            t0 = time.time()
            g, roles = adapter.build_part(pi, n, mode, outdir)
            g.save()
            graphs[g.name] = {"part": pi, "n": n, "tensors": roles}
            entry["graphs"][str(n)] = g.name
            if n == ar[0]:
                for k in requant:
                    requant[k] += g.requant[k]
            log(f"  {g.name}: {len(g.nodes)} узлов, {time.time() - t0:.0f} с")
            del g
            gc.collect()
        parts.append(entry)
    return {"graphs": graphs, "parts": parts, "requant": requant}


def ort_runner(graph_dir):
    """run(graph, feeds) на onnxruntime CPU; держит сессии одного размера шага (AR-N) — экономия памяти."""
    import onnxruntime as ort
    sess = {}

    def run(name, feeds):
        if name not in sess:
            suffix = name.split("_ar")[1]
            for k in [k for k in sess if k.split("_ar")[1] != suffix]:
                del sess[k]
            gc.collect()
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            sess[name] = ort.InferenceSession(os.path.join(graph_dir, name + ".onnx"), so, providers=["CPUExecutionProvider"])
        s = sess[name]
        names = [o.name for o in s.get_outputs()]
        return dict(zip(names, s.run(None, {i.name: feeds[i.name].astype(np.float32) for i in s.get_inputs()})))

    return run


def compare(adapter, meta, root, graph_dir, prompt_ids, steps, log=print):
    """Эталон (исходный ONNX) vs статические части: жадные токены эталона подаются обеим моделям
    (teacher forcing), сравниваются логиты на каждом шаге. -> dict метрик."""
    t0 = time.time()
    ref_logits, ref_tokens = adapter.reference_run(prompt_ids, steps)
    log(f"  эталон: {time.time() - t0:.0f} с")
    gc.collect()
    host = Host(meta, root, ort_runner(graph_dir))
    t0 = time.time()
    lg = host.prefill(prompt_ids)
    rows = []
    for step in range(steps + 1):
        r = ref_logits[step].astype(np.float64)
        p = np.exp(r - r.max()); p /= p.sum()
        q = np.exp(lg - lg.max()); q /= q.sum()
        kl = float((p * (np.log(p + 1e-30) - np.log(q + 1e-30))).sum())
        rows.append({"kl": kl, "top1": int(np.argmax(lg)) == int(ref_tokens[step]), "max_abs": float(np.abs(lg - r).max())})
        if step < steps:
            lg = host.decode(int(ref_tokens[step]))
    log(f"  статические части: {time.time() - t0:.0f} с")
    res = {"steps": len(rows), "top1": sum(r["top1"] for r in rows), "mean_kl": float(np.mean([r["kl"] for r in rows])),
           "max_kl": float(np.max([r["kl"] for r in rows])), "ref_tokens": ref_tokens.tolist(), "rows": rows}
    return res


def write_meta(path, meta):
    with open(path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
