"""Веса onnx-community q4: MatMulNBits / GatherBlockQuantized (4 бита, блоки вдоль K) и перевод в сетку HTP.

HTP v75 исполняет FullyConnected «FP16 × INT4» только с симметричными знаковыми блоками (q ∈ [-8, 7],
положительный scale, энкодинг PER_BLOCK). Блоки, которые так представимы точно, переносятся без потерь,
остальные переквантуются с подбором масштаба по минимуму MSE к весам q4.
"""
import numpy as np
import onnx
from onnx import numpy_helper as nh


def unpack_u4(packed, count):
    """uint8 [..., ceil(count/2)] -> uint8 [..., count]; младший полубайт — первый элемент."""
    lo = packed & 0x0F
    hi = packed >> 4
    out = np.stack([lo, hi], axis=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return out[..., :count]


class Q4Model:
    """Инициализаторы ONNX (с external data) и описание MatMulNBits / GatherBlockQuantized по имени узла."""

    def __init__(self, onnx_path):
        m = onnx.load(onnx_path, load_external_data=True)
        self.graph = m.graph
        self.t = {i.name: i for i in m.graph.initializer}
        self.nodes = {n.name: n for n in m.graph.node}
        self.consumers = {}
        for n in m.graph.node:
            for i in n.input:
                self.consumers.setdefault(i, []).append(n)

    def array(self, name):
        return nh.to_array(self.t[name])

    def has(self, name):
        return name in self.t

    @staticmethod
    def attrs(node):
        return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}

    def matmul_nbits(self, node_name):
        """-> (q int16 [N, K] без вычета zero point, zp int16 [N, K/bs], scale float32 [N, K/bs], block_size).
        Вес W[n, k] = (q[n, k] - zp[n, k // bs]) * scale[n, k // bs]; без входа zero point zp = 8."""
        n = self.nodes[node_name]
        a = self.attrs(n)
        assert n.op_type == "MatMulNBits" and a["bits"] == 4, node_name
        K, N, bs = a["K"], a["N"], a["block_size"]
        nb = (K + bs - 1) // bs
        q = unpack_u4(self.array(n.input[1]).reshape(N, nb * bs // 2), K).astype(np.int16)
        s = self.array(n.input[2]).astype(np.float32).reshape(N, nb)
        if len(n.input) > 3 and n.input[3]:
            zp = unpack_u4(self.array(n.input[3]).reshape(N, -1), nb).astype(np.int16)
        else:
            zp = np.full((N, nb), 8, np.int16)
        return q, zp, s, bs

    def gather_block_quantized(self, node_name):
        """GatherBlockQuantized (таблица эмбеддингов) -> (packed uint8 [rows, dim/2], scale [rows, dim/bs],
        packed zp uint8 [rows, dim/bs/2] или None, dim, block_size)."""
        n = self.nodes[node_name]
        a = self.attrs(n)
        assert a.get("bits", 4) == 4 and a.get("gather_axis", 0) == 0 and a.get("quantize_axis", 1) == 1, node_name
        packed = self.array(n.input[0])
        scales = self.array(n.input[2]).astype(np.float32)
        zp = self.array(n.input[3]) if len(n.input) > 3 and n.input[3] else None
        return packed, scales, zp, packed.shape[1] * 2, a.get("block_size", 32)


def dequant(q, zp, s, bs):
    n, k = q.shape
    nb = s.shape[1]
    w = (q.reshape(n, nb, bs).astype(np.float32) - zp[..., None]) * s[..., None]
    return w.reshape(n, k)


def to_symmetric(q, zp, s, bs, grid=13):
    """Блоки q4 (любые zp и знак scale) -> симметричная сетка HTP: (qs int8 [N, K] в [-8, 7], ss float32 [N, nb] > 0).
    Точно переносятся блоки, у которых (q - zp) * sign(s) укладывается в [-8, 7]; для остальных масштаб
    подбирается среди |s| * f, f ∈ [1, max|v|/7] (grid шагов), по минимуму MSE к весам q4."""
    n, k = q.shape
    nb = s.shape[1]
    v = (q.reshape(n, nb, bs) - zp[..., None]) * np.where(s < 0, -1, 1)[..., None].astype(np.int16)
    ss = np.abs(s).astype(np.float32)
    exact = (v.min(-1) >= -8) & (v.max(-1) <= 7)
    qs = np.where(exact[..., None], v, 0).astype(np.int16)
    bad = ~exact
    if bad.any():
        w = v[bad].astype(np.float32) * ss[bad][:, None]  # точные веса q4 этих блоков
        amax = np.abs(w).max(-1)
        base = np.where(ss[bad] > 0, ss[bad], 1e-12)
        fmax = np.maximum(amax / 7.0 / base, 1.0)
        best_err = np.full(base.shape, np.inf, np.float32)
        best_q = np.zeros(w.shape, np.int16)
        best_s = base.copy()
        for t in np.linspace(0.0, 1.0, grid):
            sc = base * (1.0 + (fmax - 1.0) * t)
            qq = np.clip(np.round(w / sc[:, None]), -8, 7)
            err = ((qq * sc[:, None] - w) ** 2).sum(-1)
            better = err < best_err
            best_err[better], best_q[better], best_s[better] = err[better], qq[better], sc[better]
        qs[bad] = best_q
        ss[bad] = best_s
    stats = {"blocks": int(n * nb), "requantized": int(bad.sum())}
    return qs.reshape(n, k).astype(np.int8), ss, stats


def block_encoding(ss, bs):
    """Энкодинг QAIRT (overrides 1.0.0) для симметричного INT4 PER_BLOCK."""
    return {"block_size": bs, "bw": 4, "dtype": "INT", "enc_type": "PER_BLOCK", "is_sym": True,
            "offset": [-8] * ss.size, "scale": ss.flatten().tolist()}


def channel_encoding_int8(scale):
    return {"bw": 8, "dtype": "INT", "enc_type": "PER_CHANNEL", "is_sym": True,
            "offset": [-128] * scale.size, "scale": scale.flatten().tolist()}
