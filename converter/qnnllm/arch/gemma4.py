"""Gemma 4 (E2B/E4B), только текст: decoder_model_merged_<quant>.onnx + embed_tokens_<quant>.onnx onnx-community.

Семантика повторяет исходный ONNX (а он — transformers Gemma4TextModel):
- слой: input_norm → внимание → post_attention_norm → +residual → pre_ff_norm → MLP (gelu_tanh(gate)·up) →
  post_ff_norm → +residual → послойный вход: gelu_tanh(gate(x))·ple[i] → proj → norm → +residual → ·layer_scalar;
- внимание без масштаба (scale = 1, q/k нормируются RMSNorm), v нормируется RMSNorm без веса;
- скользящие слои (окно W) — RoPE local, глобальные — RoPE global (head_dim 512, partial rotary в таблице);
- слои с общим KV (последние num_kv_shared_layers) берут K/V последнего не-общего слоя того же типа (донора);
- ple = (norm(proj(inputs_embeds)·H^-½).reshape(L, P) + per_layer_inputs)·2^-½;
- logits = 30·tanh(lm_head(norm(x))/30).

KV-кэши: глобальные слои — слоты по позиции (P = CL − N на граф AR-N), скользящие — кольцо на W слотов
(слот = позиция mod W). Маски и RoPE считает хост (см. qnnllm/host.py).
"""
import json
import os
import re

import numpy as np

from ..graph import Graph
from ..q4 import Q4Model

MODEL_TYPES = ("gemma4", "gemma4_text")


class Gemma4Text:
    name = "gemma4_text"

    def __init__(self, repo_dir, quant, context_length=4096, parts=None):
        self.dir = repo_dir
        self.quant = quant
        cfg = json.load(open(os.path.join(repo_dir, "config.json")))
        self.full_config = cfg
        c = cfg.get("text_config", cfg)
        self.c = c
        self.L = c["num_hidden_layers"]
        self.H = c["hidden_size"]
        self.nh = c["num_attention_heads"]
        self.kvh = c["num_key_value_heads"]
        self.hd = c["head_dim"]
        self.ghd = c.get("global_head_dim") or self.hd
        self.gkvh = c.get("num_global_key_value_heads") or self.kvh
        self.P = c["hidden_size_per_layer_input"]
        self.W = c["sliding_window"]
        self.eps = c["rms_norm_eps"]
        self.softcap = c.get("final_logit_softcapping")
        self.types = c["layer_types"]
        self.vocab = c["vocab_size"]
        self.cl = context_length
        first_shared = self.L - c.get("num_kv_shared_layers", 0)
        self.first_shared = first_shared
        prev = self.types[:first_shared]
        # донор для общих слоёв — последний не-общий слой того же типа
        self.donor = {t: max(i for i, x in enumerate(prev) if x == t) for t in set(prev)}
        assert not c.get("enable_moe_block") and not c.get("attention_k_eq_v") and not c.get("use_double_wide_mlp")
        self.parts = parts or self.default_parts()
        self.dec = None
        self.emb = None

    # --- файлы и разбиение -------------------------------------------------------------------------

    @staticmethod
    def download_patterns(quant):
        return [f"onnx/decoder_model_merged_{quant}.onnx*", f"onnx/embed_tokens_{quant}.onnx*", "config.json",
                "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                "chat_template.jinja", "chat_template.json", "README.md", "LICENSE*"]

    def default_parts(self):
        # ≈ равные части по 8–9 слоёв (≈0,45–0,55 ГБ на context binary для E4B)
        n = max(1, round(self.L / 8.5))
        bounds = np.linspace(0, self.L, n + 1).round().astype(int)
        return [list(range(bounds[i], bounds[i + 1])) for i in range(n)]

    def load(self):
        if self.dec is None:
            self.dec = Q4Model(os.path.join(self.dir, "onnx", f"decoder_model_merged_{self.quant}.onnx"))

    def is_global(self, i):
        return self.types[i] == "full_attention"

    def kv_layer(self, i):
        """Слой, чей KV-кэш использует слой i (сам или донор)."""
        return i if i < self.first_shared else self.donor[self.types[i]]

    def kv_dims(self, i):
        return (self.gkvh, self.ghd) if self.is_global(i) else (self.kvh, self.hd)

    def cache_group(self, i):
        return "global" if self.is_global(i) else "local"

    def past_slots(self, group, n):
        return self.cl - n if group == "global" else self.W

    # --- граф части ---------------------------------------------------------------------------------

    def _w(self, path):
        return self.dec.matmul_nbits(f"/model/{path}/MatMul_Quant")

    def _vec(self, name):
        return self.dec.array(name).astype(np.float32)

    def build_part(self, pi, n, mode, outdir):
        """-> (Graph, roles): roles — описание входов/выходов для metadata.json."""
        self.load()
        layers = self.parts[pi]
        last = pi == len(self.parts) - 1
        g = Graph(f"p{pi}_ar{n}", mode, outdir)
        roles = []
        H, P, L = self.H, self.P, self.L
        x = g.inp("hidden_in", [1, n, H])
        roles.append({"name": "hidden_in", "role": "embedding", "source": "tok"} if pi == 0 else
                     {"name": "hidden_in", "role": "carry"})
        if pi == 0:
            pin = g.inp("ple_tok", [1, n, L * P])
            roles.append({"name": "ple_tok", "role": "embedding", "source": "ple"})
            q, zp, s, bs = self._w("per_layer_projection")
            proj = g.op("Mul", [g.linear_q4(x, "per_layer_projection.weight", q, zp, s, bs), g.scalar(self.H ** -0.5)])
            proj = g.op("Reshape", [proj, g.ints([1, n, L, P])])
            proj = g.rmsnorm(proj, self._vec("model.per_layer_projection_norm.weight"), self.eps, wname="model.per_layer_projection_norm.weight")
            tok = g.op("Reshape", [pin, g.ints([1, n, L, P])])
            ple = g.op("Mul", [g.op("Add", [proj, tok]), g.scalar(2 ** -0.5)])
        else:
            ple = g.inp("ple", [1, n, L, P])
            roles.append({"name": "ple", "role": "carry"})
        masks, ropes = {}, {}

        def mask(group):
            if group not in masks:
                t = self.past_slots(group, n) + n
                masks[group] = g.inp(f"mask_{group}", [1, 1, n, t])
                roles.append({"name": f"mask_{group}", "role": "mask", "cache_group": group,
                              "window": self.W if group == "local" else 0})
            return masks[group]

        def rope(group):
            if group not in ropes:
                d = self.ghd if group == "global" else self.hd
                ropes[group] = (g.inp(f"rope_cos_{group}", [1, n, 1, d]), g.inp(f"rope_sin_{group}", [1, n, 1, d]))
                roles.append({"name": f"rope_cos_{group}", "role": "rope_cos", "table": group})
                roles.append({"name": f"rope_sin_{group}", "role": "rope_sin", "table": group})
            return ropes[group]

        kv_all = {}  # слой-владелец кэша -> (k_all, v_all) в этой части

        def kv_for(owner, new_k=None, new_v=None):
            if owner in kv_all:
                return kv_all[owner]
            kvh, hd = self.kv_dims(owner)
            group = self.cache_group(owner)
            ps = self.past_slots(group, n)
            pk = g.inp(f"past_key_{owner}", [1, kvh, hd, ps])
            pv = g.inp(f"past_value_{owner}", [1, kvh, ps, hd])
            roles.append({"name": f"past_key_{owner}", "role": "cache_in", "cache": f"k{owner}"})
            roles.append({"name": f"past_value_{owner}", "role": "cache_in", "cache": f"v{owner}"})
            if new_k is None:  # донор в другой части: его новые K/V приходят переносом
                new_k = g.inp(f"new_key_{owner}", [1, kvh, hd, n])
                new_v = g.inp(f"new_value_{owner}", [1, kvh, n, hd])
                roles.append({"name": f"new_key_{owner}", "role": "carry"})
                roles.append({"name": f"new_value_{owner}", "role": "carry"})
            kv_all[owner] = (g.op("Concat", [pk, new_k], axis=3), g.op("Concat", [pv, new_v], axis=2))
            return kv_all[owner]

        for i in layers:
            pre = f"model.layers.{i}"
            glob = self.is_global(i)
            hd = self.ghd if glob else self.hd
            group = self.cache_group(i)
            cos, sin = rope(group)
            res = x
            xn = g.rmsnorm(x, self._vec(f"{pre}.input_layernorm.weight"), self.eps, wname=f"{pre}.input_layernorm.weight")
            q = g.linear_q4(xn, f"{pre}.q_proj.weight", *self._w(f"layers.{i}/attn/q_proj"))
            q = g.op("Reshape", [q, g.ints([1, n, self.nh, hd])])
            q = g.rmsnorm(q, self._vec(f"{pre}.attn.q_norm.layernorm.weight"), self.eps, wname=f"{pre}.attn.q_norm.layernorm.weight")
            q = g.rope(q, cos, sin, hd)
            if i < self.first_shared:
                kvh, _ = self.kv_dims(i)
                k = g.linear_q4(xn, f"{pre}.k_proj.weight", *self._w(f"layers.{i}/attn/k_proj"))
                v = g.linear_q4(xn, f"{pre}.v_proj.weight", *self._w(f"layers.{i}/attn/v_proj"))
                k = g.rmsnorm(g.op("Reshape", [k, g.ints([1, n, kvh, hd])]),
                              self._vec(f"{pre}.attn.k_norm.layernorm.weight"), self.eps, wname=f"{pre}.attn.k_norm.layernorm.weight")
                k = g.rope(k, cos, sin, hd)
                v = g.rmsnorm(g.op("Reshape", [v, g.ints([1, n, kvh, hd])]), None, self.eps)
                nk = g.out(g.op("Transpose", [k], perm=[0, 2, 3, 1]), f"new_key_{i}", [1, kvh, hd, n])
                nv = g.out(g.op("Transpose", [v], perm=[0, 2, 1, 3]), f"new_value_{i}", [1, kvh, n, hd])
                roles.append({"name": f"new_key_{i}", "role": "cache_out", "cache": f"k{i}"})
                roles.append({"name": f"new_value_{i}", "role": "cache_out", "cache": f"v{i}"})
                k_all, v_all = kv_for(i, nk, nv)
            else:
                k_all, v_all = kv_for(self.kv_layer(i))
            kvh, _ = self.kv_dims(i)
            a = g.attention(q, k_all, v_all, mask(group), self.nh, kvh, n, hd, 1.0)
            a = g.linear_q4(a, f"{pre}.o_proj.weight", *self._w(f"layers.{i}/attn/o_proj"))
            a = g.rmsnorm(a, self._vec(f"{pre}.post_attention_layernorm.weight"), self.eps, wname=f"{pre}.post_attention_layernorm.weight")
            x = g.op("Add", [res, a])
            res = x
            xn = g.rmsnorm(x, self._vec(f"{pre}.pre_feedforward_layernorm.weight"), self.eps, wname=f"{pre}.pre_feedforward_layernorm.weight")
            gu = g.linear_q4(xn, f"{pre}.gate_up_proj.weight", *self._w(f"layers.{i}/mlp/gate_up_proj"))
            ff = self.c["intermediate_size"]
            gate = g.op("Slice", [gu, g.ints([0]), g.ints([ff]), g.ints([2])])
            up = g.op("Slice", [gu, g.ints([ff]), g.ints([2 * ff]), g.ints([2])])
            m = g.linear_q4(g.op("Mul", [g.gelu_tanh(gate), up]), f"{pre}.down_proj.weight",
                            *self._w(f"layers.{i}/mlp/down_proj"))
            m = g.rmsnorm(m, self._vec(f"{pre}.post_feedforward_layernorm.weight"), self.eps, wname=f"{pre}.post_feedforward_layernorm.weight")
            x = g.op("Add", [res, m])
            res = x
            pg = g.linear_q4(x, f"{pre}.per_layer_input_gate.weight", *self._w(f"layers.{i}/per_layer/per_layer_input_gate"))
            pli = g.op("Reshape", [g.op("Slice", [ple, g.ints([i]), g.ints([i + 1]), g.ints([2])]), g.ints([1, n, P])])
            pp = g.linear_q4(g.op("Mul", [g.gelu_tanh(pg), pli]), f"{pre}.per_layer_projection.weight",
                             *self._w(f"layers.{i}/per_layer/per_layer_projection"))
            pp = g.rmsnorm(pp, self._vec(f"{pre}.post_per_layer_input_norm.weight"), self.eps, wname=f"{pre}.post_per_layer_input_norm.weight")
            x = g.op("Mul", [g.op("Add", [res, pp]), g.const(f"{pre}.layer_scalar", self._vec(f"{pre}.layer_scalar"))])

        # переносы в следующие части: hidden, ple, новые K/V доноров, нужных дальше
        if not last:
            g.out(x, "hidden_out", [1, n, H])
            roles.append({"name": "hidden_out", "role": "carry_out", "to": "hidden_in"})
            g.out(ple, "ple_out", [1, n, L, P])
            roles.append({"name": "ple_out", "role": "carry_out", "to": "ple"})
            later = [j for p in self.parts[pi + 1:] for j in p]
            for owner in sorted({self.kv_layer(j) for j in later if j >= self.first_shared}):
                if owner in layers:
                    kvh, hd = self.kv_dims(owner)
                    g.out(f"new_key_{owner}", f"carry_key_{owner}", [1, kvh, hd, n])
                    g.out(f"new_value_{owner}", f"carry_value_{owner}", [1, kvh, n, hd])
                    roles.append({"name": f"carry_key_{owner}", "role": "carry_out", "to": f"new_key_{owner}"})
                    roles.append({"name": f"carry_value_{owner}", "role": "carry_out", "to": f"new_value_{owner}"})
        else:
            xl = g.op("Slice", [x, g.ints([n - 1]), g.ints([n]), g.ints([1])]) if n > 1 else x
            xl = g.rmsnorm(xl, self._vec(f"model.layers.{self.L}.final_norm_layernorm.weight"), self.eps, wname=f"model.layers.{self.L}.final_norm_layernorm.weight")
            lg = g.linear_q4(xl, "lm_head.weight", *self.dec.matmul_nbits("/lm_head/MatMul_Quant"))
            if self.softcap:
                lg = g.op("Mul", [g.op("Tanh", [g.op("Mul", [lg, g.scalar(1.0 / self.softcap)])]), g.scalar(self.softcap)])
            g.out(lg, "logits", [1, 1, self.vocab])
            roles.append({"name": "logits", "role": "logits"})
        return g, roles

    # --- данные хоста ---------------------------------------------------------------------------------

    def host_assets(self, outdir):
        """Таблицы эмбеддингов (q4 как в исходнике) и RoPE на CL позиций -> описания для metadata.json."""
        self.load()
        emb = Q4Model(os.path.join(self.dir, "onnx", f"embed_tokens_{self.quant}.onnx"))
        tables = []
        for name, node, mul_node in (("tok", "/model/embed_tokens/Gather_Quant", "/model/embed_tokens/Scale"),
                                     ("ple", "/model/embed_tokens_per_layer/Gather_Quant", "/model/embed_tokens_per_layer/Scale")):
            packed, scales, zp, dim, bs = emb.gather_block_quantized(node)
            mul = float(emb.array(emb.nodes[mul_node].input[1]))
            base = f"embed_{name}"
            packed.tofile(os.path.join(outdir, base + ".q4"))
            scales.astype(np.float16).tofile(os.path.join(outdir, base + ".scale.f16"))
            entry = {"name": name, "rows": int(packed.shape[0]), "dim": int(dim), "block_size": int(bs),
                     "format": "q4_block", "file": base + ".q4", "scales": base + ".scale.f16", "multiplier": mul}
            if zp is not None:
                zp.tofile(os.path.join(outdir, base + ".zp.u4"))
                entry["zero_points"] = base + ".zp.u4"
            tables.append(entry)
        # мультимодальные спец-токены (картинка/аудио) в послойной таблице заменяются на 0, как в исходном графе
        tables[1]["remap_to_zero"] = [int(self.full_config[k]) for k in ("image_token_id", "audio_token_id")
                                      if k in self.full_config]
        ropes = []
        for group, src in (("local", "cos_cache_local"), ("global", "cos_cache_global")):
            cos = self.dec.array(src)[: self.cl].astype(np.float32)
            sin = self.dec.array(src.replace("cos", "sin"))[: self.cl].astype(np.float32)
            np.stack([cos, sin]).astype(np.float32).tofile(os.path.join(outdir, f"rope_{group}.f32"))
            ropes.append({"name": group, "file": f"rope_{group}.f32", "positions": self.cl, "half_dim": int(cos.shape[1])})
        caches = []
        for i in range(self.first_shared):
            kvh, hd = self.kv_dims(i)
            group = self.cache_group(i)
            caches.append({"name": f"k{i}", "group": group, "layout": "BHDT", "heads": kvh, "dim": hd})
            caches.append({"name": f"v{i}", "group": group, "layout": "BHTD", "heads": kvh, "dim": hd})
        groups = [{"name": "global", "slots": "context"}, {"name": "local", "slots": self.W, "ring": True}]
        return {"embeddings": tables, "rope": ropes, "caches": caches, "cache_groups": groups}

    @staticmethod
    def chat_format():
        """Шаблон диалога (chat_template.jinja Gemma 4, без инструментов и рассуждений)."""
        return {"bos": "<bos>", "system": "<|turn>system\n{content}<turn|>\n", "user": "<|turn>user\n{content}<turn|>\n",
                "assistant": "<|turn>model\n", "end_of_turn": "<turn|>\n"}

    def special_tokens(self):
        gen = {}
        p = os.path.join(self.dir, "generation_config.json")
        if os.path.exists(p):
            gen = json.load(open(p))
        eos = gen.get("eos_token_id", self.c.get("eos_token_id"))
        eos = eos if isinstance(eos, list) else [eos]
        return {"bos": gen.get("bos_token_id", self.c.get("bos_token_id")), "eos": eos}

    # --- эталон ---------------------------------------------------------------------------------------

    def reference_session(self):
        """Исходный ONNX в onnxruntime: (embed, decoder) — для сверки."""
        import onnxruntime as ort
        so = ort.SessionOptions()
        emb = ort.InferenceSession(os.path.join(self.dir, "onnx", f"embed_tokens_{self.quant}.onnx"), so, providers=["CPUExecutionProvider"])
        dec = ort.InferenceSession(os.path.join(self.dir, "onnx", f"decoder_model_merged_{self.quant}.onnx"), so, providers=["CPUExecutionProvider"])
        return emb, dec

    def reference_run(self, ids, gen_steps):
        """Жадная генерация исходной моделью: -> (logits[steps+1, V], tokens[steps+1])."""
        emb, dec = self.reference_session()
        feeds = {}
        for i in dec.get_inputs():
            if i.name.startswith("past_key_values"):
                shape = [1 if isinstance(d, str) and "batch" in d else 0 if isinstance(d, str) else d for d in i.shape]
                feeds[i.name] = np.zeros(shape, np.float32)
        outs = [o.name for o in dec.get_outputs()]
        cur, total, logits, toks = list(ids), 0, [], []
        for _ in range(gen_steps + 1):
            e, ple = emb.run(None, {"input_ids": np.array([cur], np.int64)})
            pos = np.arange(total, total + len(cur), dtype=np.int64)[None]
            total += len(cur)
            f = dict(feeds, inputs_embeds=e, per_layer_inputs=ple, attention_mask=np.ones((1, total), np.int64),
                     position_ids=pos, num_logits_to_keep=np.array(1, np.int64))
            r = dict(zip(outs, dec.run(None, {k: v for k, v in f.items() if k in {i.name for i in dec.get_inputs()}})))
            lg = r["logits"][0, -1]
            logits.append(lg)
            for k in feeds:
                feeds[k] = r[k.replace("past_key_values", "present")]
            t = int(np.argmax(lg))
            toks.append(t)
            cur = [t]
        return np.stack(logits), np.array(toks)
