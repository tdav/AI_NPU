"""Построение статического ONNX-графа части модели: стандартные операции opset 18, веса float32 на сетке INT4
(+ энкодинги QAIRT в <граф>.encodings), остальное конвертер переводит в FP16."""
import json
import math
import os

import numpy as np
import onnx
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh

from . import q4 as q4mod


class Graph:
    def __init__(self, name, mode, outdir):
        """mode: 'exact' — точные веса q4 (сверка логики графа в ORT), 'npu' — сетка HTP + энкодинги.
        Крупные константы сразу пишутся во внешний файл <outdir>/<name>.onnx.data: protobuf ограничен 2 ГБ,
        а держать веса части в памяти незачем."""
        self.name, self.mode, self.outdir = name, mode, outdir
        os.makedirs(outdir, exist_ok=True)
        self._data = None
        self._offset = 0
        self.nodes, self.inits, self.inputs, self.outputs, self.encodings = [], [], [], [], []
        self.input_names, self.output_names = [], []
        self.requant = {"blocks": 0, "requantized": 0}
        self._n = 0
        self._consts = {}

    def _name(self, base):
        self._n += 1
        return f"{base}_{self._n}"

    def inp(self, name, shape):
        if name not in self.input_names:
            self.inputs.append(h.make_tensor_value_info(name, T.FLOAT, list(shape)))
            self.input_names.append(name)
        return name

    def out(self, tensor, name, shape):
        self.op("Identity", [tensor], out=name)
        self.outputs.append(h.make_tensor_value_info(name, T.FLOAT, list(shape)))
        self.output_names.append(name)
        return name

    def const(self, name, arr):
        arr = np.ascontiguousarray(arr)
        if arr.nbytes < (1 << 20):
            self.inits.append(nh.from_array(arr, name))
            return name
        if self._data is None:
            self._data = open(os.path.join(self.outdir, self.name + ".onnx.data"), "wb")
        pad = (-self._offset) % 4096
        self._data.write(b"\0" * pad)
        self._offset += pad
        self._data.write(arr.tobytes())
        t = onnx.TensorProto()
        t.name = name
        t.data_type = h.np_dtype_to_tensor_dtype(arr.dtype)
        t.dims.extend(arr.shape)
        t.data_location = onnx.TensorProto.EXTERNAL
        for k, v in (("location", self.name + ".onnx.data"), ("offset", str(self._offset)), ("length", str(arr.nbytes))):
            e = t.external_data.add()
            e.key, e.value = k, v
        self._offset += arr.nbytes
        self.inits.append(t)
        return name

    def scalar(self, value):
        key = ("f", float(value))
        if key not in self._consts:
            self._consts[key] = self.const(self._name("c"), np.array(value, np.float32))
        return self._consts[key]

    def ints(self, vals):
        key = ("i",) + tuple(vals)
        if key not in self._consts:
            self._consts[key] = self.const(self._name("i64"), np.array(vals, np.int64))
        return self._consts[key]

    def op(self, typ, ins, out=None, **attrs):
        out = out or self._name(typ.lower())
        self.nodes.append(h.make_node(typ, ins, [out], name=out, **attrs))
        return out

    # --- составные блоки -------------------------------------------------------------------------

    def linear_q4(self, x, wname, q, zp, s, bs):
        """x @ W^T для блоков q4 [N, K]. exact: точные веса; npu: симметричная сетка + PER_BLOCK энкодинг."""
        if self.mode == "npu":
            qs, ss, st = q4mod.to_symmetric(q, zp, s, bs)
            self.requant["blocks"] += st["blocks"]
            self.requant["requantized"] += st["requantized"]
            n, k = qs.shape
            w = (qs.reshape(n, -1, bs).astype(np.float32) * ss[..., None]).reshape(n, k)
            self.encodings.append(dict(q4mod.block_encoding(ss, bs), name=wname))
        else:
            w = q4mod.dequant(q, zp, s, bs)
        self.const(wname, np.ascontiguousarray(w.T))
        return self.op("MatMul", [x, wname])

    def linear_int8_channel(self, x, wname, w):
        """x @ W^T, W [N, K] float32: npu — INT8 per-channel (для lm_head из точных весов)."""
        if self.mode == "npu":
            sc = np.maximum(np.abs(w).max(1), 1e-12) / 127.0
            w = np.clip(np.round(w / sc[:, None]), -127, 127) * sc[:, None]
            self.encodings.append(dict(q4mod.channel_encoding_int8(sc), name=wname))
        self.const(wname, np.ascontiguousarray(w.T.astype(np.float32)))
        return self.op("MatMul", [x, wname])

    def rmsnorm(self, x, weight, eps, prescale=1.0, wname=None):
        """x * (mean(x²) + eps)^-0.5 * weight. prescale < 1 делит x перед возведением в квадрат (FP16 на HTP
        переполняется при |x| > 256): mean((x·c)²) = c²·mean(x²), поэтому eps масштабируется тем же c²."""
        xs = self.op("Mul", [x, self.scalar(prescale)]) if prescale != 1.0 else x
        var = self.op("ReduceMean", [self.op("Mul", [xs, xs]), self.ints([-1])], keepdims=1)
        inv = self.op("Reciprocal", [self.op("Sqrt", [self.op("Add", [var, self.scalar(eps * prescale * prescale)])])])
        y = self.op("Mul", [xs, inv])
        if weight is not None:
            y = self.op("Mul", [y, self.const(wname or self._name("norm_w"), weight.astype(np.float32))])
        return y

    def gelu_tanh(self, x):
        """0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))) — как Gelu(approximate='tanh')."""
        x3 = self.op("Mul", [self.op("Mul", [x, x]), x])
        inner = self.op("Mul", [self.op("Add", [x, self.op("Mul", [x3, self.scalar(0.044715)])]),
                                self.scalar(math.sqrt(2.0 / math.pi))])
        return self.op("Mul", [self.op("Mul", [x, self.scalar(0.5)]),
                               self.op("Add", [self.op("Tanh", [inner]), self.scalar(1.0)])])

    def silu(self, x):
        return self.op("Mul", [x, self.op("Sigmoid", [x])])

    def rope(self, x, cos, sin, dim, axis=-1):
        """x·cos + rotate_half(x)·sin; cos/sin полной ширины dim (половины продублированы)."""
        x1 = self.op("Slice", [x, self.ints([0]), self.ints([dim // 2]), self.ints([axis])])
        x2 = self.op("Slice", [x, self.ints([dim // 2]), self.ints([dim]), self.ints([axis])])
        rot = self.op("Concat", [self.op("Neg", [x2]), x1], axis=axis)
        return self.op("Add", [self.op("Mul", [x, cos]), self.op("Mul", [rot, sin])])

    def attention(self, q, k_all, v_all, mask, n_q_heads, n_kv_heads, n, head_dim, scale):
        """GQA: q [1, N, Hq, D] (после RoPE), k_all [1, Hkv, D, T], v_all [1, Hkv, T, D], mask [1, 1, N, T]
        аддитивная. Головы q g·r..g·r+r-1 -> kv-голова g. -> [1, N, Hq·D]."""
        r = n_q_heads // n_kv_heads
        qt = self.op("Transpose", [q], perm=[0, 2, 1, 3])                       # [1, Hq, N, D]
        qg = self.op("Reshape", [qt, self.ints([1, n_kv_heads, r * n, head_dim])])
        if scale != 1.0:
            qg = self.op("Mul", [qg, self.scalar(scale)])
        sc = self.op("MatMul", [qg, k_all])                                     # [1, Hkv, rN, T]
        m = mask if r == 1 else self.op("Concat", [mask] * r, axis=2) if n > 1 else self.op("Tile", [mask, self.ints([1, 1, r, 1])])
        pr = self.op("Softmax", [self.op("Add", [sc, m])], axis=-1)
        o = self.op("MatMul", [pr, v_all])                                      # [1, Hkv, rN, D]
        o = self.op("Reshape", [o, self.ints([1, n_q_heads, n, head_dim])])
        o = self.op("Transpose", [o], perm=[0, 2, 1, 3])
        return self.op("Reshape", [o, self.ints([1, n, n_q_heads * head_dim])])

    # --- сохранение --------------------------------------------------------------------------------

    def model(self):
        g = h.make_graph(self.nodes, self.name, self.inputs, self.outputs, self.inits)
        m = h.make_model(g, opset_imports=[h.make_opsetid("", 18)], producer_name="qnnllm")
        m.ir_version = 8
        return m

    def save(self):
        if self._data is not None:
            self._data.close()
            self._data = None
        path = os.path.join(self.outdir, self.name + ".onnx")
        onnx.save(self.model(), path)
        if self.encodings:
            with open(os.path.join(self.outdir, self.name + ".encodings"), "w") as f:
                json.dump({"version": "1.0.0", "activation_encodings": [], "param_encodings": self.encodings}, f)
        return path
