"""Хостовая логика инференса модели qnn-llm по metadata.json (эталон для C++-рантайма native/lfm2-npu).

Шаг: N токенов (AR-N граф, левый паддинг до N) прогоняется через все части по порядку. Роли входов:
  embedding  — строки таблицы эмбеддингов (q4_block с zero point или f16), ×multiplier; паддинг — нули;
  carry      — тензор, выданный раньше выходом carry_out другой части (hidden, ple, K/V донора);
  mask       — аддитивная маска [1,1,N,S+N] по позициям слотов группы кэша (0 / -10000), окно window;
  rope_cos/rope_sin — строки таблицы RoPE по позициям, половины продублированы: [1,N,1,2·half];
  valid      — [1,N,1]: 0 у паддинга;
  cache_in   — прошлое кэша в раскладке графа (BHDT: [1,H,D,S], BHTD: [1,H,S,D]);
  state_in   — рекуррентное состояние (conv LFM2).
Выходы: cache_out (новые K/V [..N..] -> слоты), carry_out, state_out, logits [1,1,V] последней строки.
Группы кэшей: slots="context" — слот = позиция, у графа AR-N S = CL − N; ring W — слот = позиция mod W, S = W.
Новые K/V пишутся в кэши после прогона всех частей (части видят прошлое шага + переносы текущего).
"""
import json
import os

import numpy as np

MASK_NEG = -10000.0


class Embedding:
    def __init__(self, d, root):
        self.d = d
        self.dim = d["dim"]
        self.mult = d.get("multiplier", 1.0)
        self.remap = set(d.get("remap_to_zero", []))
        rows = d["rows"]
        if d["format"] == "f16":
            self.table = np.memmap(os.path.join(root, d["file"]), np.float16, "r", shape=(rows, self.dim))
        else:
            bs = d["block_size"]
            self.bs = bs
            self.packed = np.memmap(os.path.join(root, d["file"]), np.uint8, "r", shape=(rows, self.dim // 2))
            self.scales = np.memmap(os.path.join(root, d["scales"]), np.float16, "r", shape=(rows, self.dim // bs))
            zp = d.get("zero_points")
            nb = self.dim // bs
            self.zp = np.memmap(os.path.join(root, zp), np.uint8, "r", shape=(rows, (nb + 1) // 2)) if zp else None

    def rows(self, ids):
        ids = np.array([0 if i in self.remap else i for i in ids], np.int64)
        if self.d["format"] == "f16":
            return self.table[ids].astype(np.float32) * self.mult
        p = self.packed[ids]
        q = np.stack([p & 0x0F, p >> 4], -1).reshape(len(ids), self.dim).astype(np.float32)
        nb = self.dim // self.bs
        if self.zp is not None:
            z = self.zp[ids]
            zp = np.stack([z & 0x0F, z >> 4], -1).reshape(len(ids), -1)[:, :nb].astype(np.float32)
        else:
            zp = np.full((len(ids), nb), 8.0, np.float32)
        s = self.scales[ids].astype(np.float32)
        w = (q.reshape(len(ids), nb, self.bs) - zp[..., None]) * s[..., None]
        return w.reshape(len(ids), self.dim) * self.mult


class Host:
    def __init__(self, meta, root, run):
        """run(graph_name, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]."""
        self.m, self.root, self.run = meta, root, run
        self.cl = meta["context_length"]
        self.ar = max(meta["ar"])
        self.emb = {e["name"]: Embedding(e, root) for e in meta["embeddings"]}
        self.rope = {}
        for r in meta.get("rope", []):
            t = np.fromfile(os.path.join(root, r["file"]), np.float32).reshape(2, r["positions"], r["half_dim"])
            self.rope[r["name"]] = (np.concatenate([t[0], t[0]], 1), np.concatenate([t[1], t[1]], 1))
        self.groups = {g["name"]: g for g in meta["cache_groups"]}
        self.caches = {c["name"]: c for c in meta["caches"]}
        self.reset()

    def slots(self, group, n):
        s = self.groups[group]["slots"]
        return self.cl - n if s == "context" else int(s)

    def reset(self):
        self.n_past = 0
        self.buf = {}  # (cache, n) -> массив в раскладке графа AR-n
        for n in self.m["ar"]:
            for c in self.caches.values():
                S = self.slots(c["group"], n)
                shape = (1, c["heads"], c["dim"], S) if c["layout"] == "BHDT" else (1, c["heads"], S, c["dim"])
                self.buf[(c["name"], n)] = np.zeros(shape, np.float32)
        self.slot_pos = {g: np.full(self.slots(g, 1) if self.groups[g]["slots"] == "context" else int(self.groups[g]["slots"]),
                                    -1, np.int64) for g in self.groups}
        self.state = {s["name"]: np.zeros(s["shape"], np.float32) for s in self.m.get("states", [])}

    def _mask(self, group, window, n, pad, pos):
        S = self.slots(group, n)
        sp = self.slot_pos[group][:S]
        m = np.full((n, S + n), MASK_NEG, np.float32)
        for r in range(n):
            if r < pad:
                m[r, S + r] = 0.0
                continue
            q = pos[r]
            ok = (sp >= 0) & (sp <= q)
            if window:
                ok &= (q - sp) < window
            m[r, :S][ok] = 0.0
            lo = max(pad, r - window + 1) if window else pad
            m[r, S + lo:S + r + 1] = 0.0
        return m[None, None]

    def step(self, tokens, n):
        real = len(tokens)
        pad = n - real
        assert 0 < real <= n and self.n_past + real <= self.cl
        assert n == 1 or self.n_past <= self.cl - n, "prefill AR-N за пределами кэша"
        pos = np.zeros(n, np.int64)
        pos[pad:] = self.n_past + np.arange(real)
        carry, writes, logits = {}, [], None
        for pi, part in enumerate(self.m["parts"]):
            gname = part["graphs"][str(n)]
            spec = self.m["graphs"][gname]
            feeds = {}
            for t in spec["tensors"]:
                role, name = t["role"], t["name"]
                if role == "embedding":
                    e = self.emb[t["source"]]
                    x = np.zeros((1, n, e.dim), np.float32)
                    x[0, pad:] = e.rows(tokens)
                    feeds[name] = x
                elif role == "carry":
                    feeds[name] = carry[name]
                elif role == "mask":
                    feeds[name] = self._mask(t["cache_group"], t.get("window", 0), n, pad, pos)
                elif role in ("rope_cos", "rope_sin"):
                    tab = self.rope[t["table"]][0 if role == "rope_cos" else 1]
                    feeds[name] = tab[pos][None, :, None, :]
                elif role == "valid":
                    v = np.zeros((1, n, 1), np.float32)
                    v[0, pad:] = 1.0
                    feeds[name] = v
                elif role == "cache_in":
                    feeds[name] = self.buf[(t["cache"], n)]
                elif role == "state_in":
                    feeds[name] = self.state[t["state"]]
            out = self.run(gname, feeds)
            for t in spec["tensors"]:
                role, name = t["role"], t["name"]
                if role == "cache_out":
                    writes.append((t["cache"], out[name]))
                elif role == "carry_out":
                    carry[t["to"]] = out[name]
                elif role == "state_out":
                    self.state[t["state"]] = out[name]
                elif role == "logits":
                    logits = out[name].reshape(-1)
        for cname, new in writes:
            c = self.caches[cname]
            group = self.groups[c["group"]]
            for r in range(pad, n):
                p = int(pos[r])
                slot = p if group["slots"] == "context" else p % int(group["slots"])
                for nn in self.m["ar"]:
                    b = self.buf[(cname, nn)]
                    if slot >= b.shape[3 if c["layout"] == "BHDT" else 2]:
                        continue
                    if c["layout"] == "BHDT":
                        b[0, :, :, slot] = new[0, :, :, r]
                    else:
                        b[0, :, slot, :] = new[0, :, r, :]
        for r in range(pad, n):
            g_done = set()
            for c in self.caches.values():
                g = c["group"]
                if g in g_done:
                    continue
                g_done.add(g)
                grp = self.groups[g]
                p = int(pos[r])
                slot = p if grp["slots"] == "context" else p % int(grp["slots"])
                if slot < len(self.slot_pos[g]):
                    self.slot_pos[g][slot] = p
        self.n_past += real
        return logits

    def prefill(self, tokens):
        tokens = list(tokens)
        first = len(tokens) % self.ar or self.ar
        logits, i = None, 0
        while i < len(tokens):
            ln = first if i == 0 else min(self.ar, len(tokens) - i)
            chunk = tokens[i:i + ln]
            if ln == 1 or self.n_past > self.cl - self.ar:
                for t in chunk:
                    logits = self.step([t], 1)
            else:
                logits = self.step(chunk, self.ar)
            i += ln
        return logits

    def decode(self, token):
        return self.step([token], 1)


def load_meta(root):
    return json.load(open(os.path.join(root, "metadata.json")))
