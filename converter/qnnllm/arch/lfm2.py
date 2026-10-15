"""LFM2 / LFM2.5 (Liquid AI), onnx-community model_<quant>.onnx: conv-блоки (короткая свёртка, ядро 3) +
GQA-внимание с qk-norm, SwiGLU. Проверено на LFM2-2.6B q4 (бакет tdavron/lfm2-2.6b-q4-qnn-sm8650-bucket).

Conv-слой: in_proj → B, C, x; Bx = B·x (у левого паддинга ×valid = 0) → окно [state(2), Bx(N)] → свёртка
3 сдвигами → y = C·conv → out_proj; новое состояние — 2 последних строки окна.
"""
import json
import os

import numpy as np

from ..graph import Graph
from ..q4 import Q4Model

MODEL_TYPES = ("lfm2",)


class Lfm2:
    name = "lfm2"

    def __init__(self, repo_dir, quant, context_length=4096, parts=None):
        self.dir, self.quant, self.cl = repo_dir, quant, context_length
        c = json.load(open(os.path.join(repo_dir, "config.json")))
        self.c = c
        self.types = c["layer_types"]
        self.L = c["num_hidden_layers"]
        self.H = c["hidden_size"]
        self.nh = c["num_attention_heads"]
        self.kvh = c["num_key_value_heads"]
        self.hd = self.H // self.nh
        self.eps = c.get("norm_eps", 1e-5)
        self.vocab = c["vocab_size"]
        n = max(1, round(self.L / 10))
        b = np.linspace(0, self.L, n + 1).round().astype(int)
        self.parts = parts or [list(range(b[i], b[i + 1])) for i in range(n)]
        self.m = None

    @staticmethod
    def download_patterns(quant):
        return [f"onnx/model_{quant}.onnx*", "config.json", "generation_config.json", "tokenizer.json",
                "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "README.md", "LICENSE*"]

    def load(self):
        if self.m is None:
            self.m = Q4Model(os.path.join(self.dir, "onnx", f"model_{self.quant}.onnx"))
            # имя узла MatMulNBits по префиксу веса: model.layers.0.conv.in_proj -> узел
            self.nodes = {}
            for n in self.m.graph.node:
                if n.op_type == "MatMulNBits":
                    self.nodes[n.input[1].replace(".MatMul.weight_Q4", "")] = n.name

    def _w(self, prefix):
        return self.m.matmul_nbits(self.nodes[prefix])

    def _vec(self, name):
        return self.m.array(name).astype(np.float32)

    def build_part(self, pi, n, mode, outdir):
        self.load()
        layers = self.parts[pi]
        last = pi == len(self.parts) - 1
        g = Graph(f"p{pi}_ar{n}", mode, outdir)
        H, roles = self.H, []
        x = g.inp("hidden_in", [1, n, H])
        roles.append({"name": "hidden_in", "role": "embedding", "source": "tok"} if pi == 0 else
                     {"name": "hidden_in", "role": "carry"})
        P = self.cl - n
        mask = g.inp("mask_context", [1, 1, n, P + n])
        roles.append({"name": "mask_context", "role": "mask", "cache_group": "context", "window": 0})
        cos = g.inp("rope_cos", [1, n, 1, self.hd])
        sin = g.inp("rope_sin", [1, n, 1, self.hd])
        roles += [{"name": "rope_cos", "role": "rope_cos", "table": "rope"}, {"name": "rope_sin", "role": "rope_sin", "table": "rope"}]
        valid = None
        if n > 1:
            valid = g.inp("valid", [1, n, 1])
            roles.append({"name": "valid", "role": "valid"})
        for i in layers:
            pre = f"model.layers.{i}"
            res = x
            nw = f"{pre}.operator_layernorm.weight"
            xn = g.rmsnorm(x, self._vec(nw), self.eps, wname=nw)
            if self.types[i] == "conv":
                st = g.inp(f"conv_state_{i}", [1, 2, H])
                roles.append({"name": f"conv_state_{i}", "role": "state_in", "state": f"conv{i}"})
                bcx = g.linear_q4(xn, f"{pre}.conv.in_proj.weight", *self._w(f"{pre}.conv.in_proj"))
                b = g.op("Slice", [bcx, g.ints([0]), g.ints([H]), g.ints([2])])
                c = g.op("Slice", [bcx, g.ints([H]), g.ints([2 * H]), g.ints([2])])
                xx = g.op("Slice", [bcx, g.ints([2 * H]), g.ints([3 * H]), g.ints([2])])
                bx = g.op("Mul", [b, xx])
                if valid is not None:
                    bx = g.op("Mul", [bx, valid])
                win = g.op("Concat", [st, bx], axis=1)
                cw = self._vec(f"{pre}.conv.conv.weight")[:, 0, :]
                acc = None
                for k in range(3):
                    sl = g.op("Slice", [win, g.ints([k]), g.ints([k + n]), g.ints([1])])
                    t = g.op("Mul", [sl, g.const(f"{pre}.conv.conv.w{k}", cw[:, k].copy())])
                    acc = t if acc is None else g.op("Add", [acc, t])
                g.out(g.op("Slice", [win, g.ints([n]), g.ints([n + 2]), g.ints([1])]), f"new_conv_state_{i}", [1, 2, H])
                roles.append({"name": f"new_conv_state_{i}", "role": "state_out", "state": f"conv{i}"})
                y = g.linear_q4(g.op("Mul", [c, acc]), f"{pre}.conv.out_proj.weight", *self._w(f"{pre}.conv.out_proj"))
            else:
                pk = g.inp(f"past_key_{i}", [1, self.kvh, self.hd, P])
                pv = g.inp(f"past_value_{i}", [1, self.kvh, P, self.hd])
                roles += [{"name": f"past_key_{i}", "role": "cache_in", "cache": f"k{i}"},
                          {"name": f"past_value_{i}", "role": "cache_in", "cache": f"v{i}"}]
                q = g.linear_q4(xn, f"{pre}.attn.q_proj.weight", *self._w(f"{pre}.attn.q_proj"))
                k = g.linear_q4(xn, f"{pre}.attn.k_proj.weight", *self._w(f"{pre}.attn.k_proj"))
                v = g.linear_q4(xn, f"{pre}.attn.v_proj.weight", *self._w(f"{pre}.attn.v_proj"))
                qn, kn = f"{pre}.attn.q_norm.layernorm.weight", f"{pre}.attn.k_norm.layernorm.weight"
                q = g.rope(g.rmsnorm(g.op("Reshape", [q, g.ints([1, n, self.nh, self.hd])]), self._vec(qn), self.eps, wname=qn), cos, sin, self.hd)
                k = g.rope(g.rmsnorm(g.op("Reshape", [k, g.ints([1, n, self.kvh, self.hd])]), self._vec(kn), self.eps, wname=kn), cos, sin, self.hd)
                v = g.op("Reshape", [v, g.ints([1, n, self.kvh, self.hd])])
                nk = g.out(g.op("Transpose", [k], perm=[0, 2, 3, 1]), f"new_key_{i}", [1, self.kvh, self.hd, n])
                nv = g.out(g.op("Transpose", [v], perm=[0, 2, 1, 3]), f"new_value_{i}", [1, self.kvh, n, self.hd])
                roles += [{"name": f"new_key_{i}", "role": "cache_out", "cache": f"k{i}"},
                          {"name": f"new_value_{i}", "role": "cache_out", "cache": f"v{i}"}]
                a = g.attention(q, g.op("Concat", [pk, nk], axis=3), g.op("Concat", [pv, nv], axis=2), mask,
                                self.nh, self.kvh, n, self.hd, self.hd ** -0.5)
                y = g.linear_q4(a, f"{pre}.attn.o_proj.weight", *self._w(f"{pre}.attn.o_proj"))
            x = g.op("Add", [y, res])
            fw = f"{pre}.ffn_layernorm.weight"
            xn = g.rmsnorm(x, self._vec(fw), self.eps, wname=fw)
            a = g.linear_q4(xn, f"{pre}.mlp.w1.weight", *self._w(f"{pre}.mlp.gate_proj"))
            b = g.linear_q4(xn, f"{pre}.mlp.w3.weight", *self._w(f"{pre}.mlp.up_proj"))
            y = g.linear_q4(g.op("Mul", [g.silu(a), b]), f"{pre}.mlp.w2.weight", *self._w(f"{pre}.mlp.down_proj"))
            x = g.op("Add", [y, x])
        if last:
            xl = g.op("Slice", [x, g.ints([n - 1]), g.ints([n]), g.ints([1])]) if n > 1 else x
            fn = f"model.layers.{self.L}.final_norm_layernorm.weight"
            xl = g.rmsnorm(xl, self._vec(fn), self.eps, wname=fn)
            g.out(g.linear_int8_channel(xl, "lm_head.weight", self._vec("model.embed_tokens.weight")), "logits", [1, 1, self.vocab])
            roles.append({"name": "logits", "role": "logits"})
        else:
            g.out(x, "hidden_out", [1, n, H])
            roles.append({"name": "hidden_out", "role": "carry_out", "to": "hidden_in"})
        return g, roles

    def host_assets(self, outdir):
        self.load()
        e = self._vec("model.embed_tokens.weight").astype(np.float16)
        e.tofile(os.path.join(outdir, "embed_tok.f16"))
        cos = self.m.array("cos_cache")[: self.cl].astype(np.float32)
        sin = self.m.array("sin_cache")[: self.cl].astype(np.float32)
        np.stack([cos, sin]).tofile(os.path.join(outdir, "rope.f32"))
        caches = []
        for i, t in enumerate(self.types):
            if t != "conv":
                caches += [{"name": f"k{i}", "group": "context", "layout": "BHDT", "heads": self.kvh, "dim": self.hd},
                           {"name": f"v{i}", "group": "context", "layout": "BHTD", "heads": self.kvh, "dim": self.hd}]
        return {"embeddings": [{"name": "tok", "rows": int(e.shape[0]), "dim": int(e.shape[1]), "format": "f16",
                                "file": "embed_tok.f16", "multiplier": 1.0}],
                "rope": [{"name": "rope", "file": "rope.f32", "positions": self.cl, "half_dim": int(cos.shape[1])}],
                "caches": caches, "cache_groups": [{"name": "context", "slots": "context"}],
                "states": [{"name": f"conv{i}", "shape": [1, 2, self.H]} for i, t in enumerate(self.types) if t == "conv"]}

    @staticmethod
    def chat_format():
        """Шаблон диалога (chat_template.jinja LFM2, без инструментов)."""
        return {"bos": "<|startoftext|>", "system": "<|im_start|>system\n{content}<|im_end|>\n",
                "user": "<|im_start|>user\n{content}<|im_end|>\n", "assistant": "<|im_start|>assistant\n",
                "end_of_turn": "<|im_end|>\n"}

    def special_tokens(self):
        gen = json.load(open(os.path.join(self.dir, "generation_config.json"))) if os.path.exists(os.path.join(self.dir, "generation_config.json")) else {}
        eos = gen.get("eos_token_id", self.c.get("eos_token_id"))
        return {"bos": gen.get("bos_token_id", self.c.get("bos_token_id")), "eos": eos if isinstance(eos, list) else [eos]}

    def reference_run(self, ids, gen_steps):
        import onnxruntime as ort
        s = ort.InferenceSession(os.path.join(self.dir, "onnx", f"model_{self.quant}.onnx"), providers=["CPUExecutionProvider"])
        feeds = {}
        for i in s.get_inputs():
            if i.name.startswith("past_conv"):
                feeds[i.name] = np.zeros((1, self.H, 3), np.float32)
            elif i.name.startswith("past_key_values"):
                feeds[i.name] = np.zeros((1, self.kvh, 0, self.hd), np.float32)
        outs = [o.name for o in s.get_outputs()]
        cur, total, logits, toks = list(ids), 0, [], []
        for _ in range(gen_steps + 1):
            total += len(cur)
            r = dict(zip(outs, s.run(None, dict(feeds, input_ids=np.array([cur], np.int64), attention_mask=np.ones((1, total), np.int64)))))
            logits.append(r["logits"][0, -1])
            for k in feeds:
                feeds[k] = r[k.replace("past_conv", "present_conv").replace("past_key_values", "present")]
            toks.append(int(np.argmax(logits[-1])))
            cur = [toks[-1]]
        return np.stack(logits), np.array(toks)
