"""Вызовы QAIRT SDK (Linux x86_64 и Windows x86_64): qairt-converter (ONNX + энкодинги -> DLC) и
qnn-context-binary-generator (DLC графов AR-N части -> один context binary с общими весами).

Ограничение QAIRT 2.40 для Windows: HtpPrepare.dll не компилирует под мобильные HTP (v75 SM8650 и др.,
«Unsupported HTP Arch»), только под Snapdragon X. Поэтому на Windows context binary собирается Linux-версией
QAIRT в WSL (--wsl-qairt); конвертация в DLC и остальные шаги идут нативно."""
import json
import os
import platform
import subprocess
import sys

# SoC -> (soc_id, dsp_arch, vtcm МБ); источник — таблица QNN HTP (docs/QNN, device_info sherpa-onnx)
SOCS = {
    "SM8550": (43, "v73", 8),   # Snapdragon 8 Gen 2
    "SM8650": (57, "v75", 8),   # Snapdragon 8 Gen 3
    "SM8750": (69, "v79", 8),   # Snapdragon 8 Elite
    "SM8850": (87, "v81", 8),   # Snapdragon 8 Elite Gen 5
}


class Qairt:
    def __init__(self, root=None, wsl_qairt=None, wsl_env=None):
        root = root or os.environ.get("QAIRT_SDK_ROOT") or os.environ.get("QNN_SDK_ROOT")
        if not root or not os.path.isdir(os.path.join(root, "lib", "python")):
            raise SystemExit("Не найден QAIRT SDK: укажите --qairt <путь к qairt/2.40.0.251030> или QNN_SDK_ROOT")
        self.root = os.path.abspath(root)
        self.windows = platform.system() == "Windows"
        self.plat = "x86_64-windows-msvc" if self.windows else "x86_64-linux-clang"
        self.bin = os.path.join(self.root, "bin", self.plat)
        self.lib = os.path.join(self.root, "lib", self.plat)
        self.version = os.path.basename(self.root.rstrip("/\\"))
        self.wsl_qairt, self.wsl_env = wsl_qairt, wsl_env

    def env(self):
        e = dict(os.environ)
        e["QNN_SDK_ROOT"] = self.root
        e["PYTHONPATH"] = os.pathsep.join(p for p in [os.path.join(self.root, "lib", "python"), e.get("PYTHONPATH")] if p)
        if self.windows:
            e["PATH"] = os.pathsep.join([self.bin, self.lib, e.get("PATH", "")])
        else:
            e["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in [self.lib, e.get("LD_LIBRARY_PATH")] if p)
        return e

    def _run(self, args, log_path):
        with open(log_path, "w") as log:
            r = subprocess.run(args, env=self.env(), stdout=log, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            tail = open(log_path, errors="replace").read()[-3000:]
            raise RuntimeError(f"{os.path.basename(args[1] if args[0] == sys.executable else args[0])} "
                               f"завершился с кодом {r.returncode}, журнал {log_path}:\n{tail}")

    def convert(self, onnx_path, encodings, dlc_path):
        args = [sys.executable, os.path.join(self.bin, "qairt-converter"), "-i", onnx_path, "-o", dlc_path,
                "--float_bitwidth", "16"]
        if encodings and os.path.exists(encodings):
            args += ["--quantization_overrides", encodings]
        self._run(args, dlc_path + ".log")

    @staticmethod
    def wsl_path(path):
        """E:\\a\\b -> /mnt/e/a/b."""
        p = os.path.abspath(path)
        return "/mnt/" + p[0].lower() + p[2:].replace("\\", "/")

    def context_binary(self, dlc_paths, graph_names, soc, out_dir, name):
        if self.windows:
            if not self.wsl_qairt:
                raise SystemExit("QAIRT для Windows не компилирует context binary под мобильные NPU (Unsupported HTP Arch). "
                                 "Укажите --wsl-qairt <путь к Linux-версии QAIRT в WSL>, например ~/qairt/qairt/2.40.0.251030.")
            return self._context_binary_wsl(dlc_paths, graph_names, soc, out_dir, name)
        soc_id, arch, vtcm = SOCS[soc]
        cfg = os.path.join(out_dir, name + ".htp.json")
        ext = os.path.join(out_dir, name + ".ext.json")
        json.dump({"graphs": [{"vtcm_mb": vtcm, "O": 3, "graph_names": graph_names}],
                   "devices": [{"device_id": 0, "soc_id": soc_id, "dsp_arch": arch,
                                "cores": [{"core_id": 0, "perf_profile": "burst", "rpc_control_latency": 100}]}],
                   "context": {"weight_sharing_enabled": len(graph_names) > 1}}, open(cfg, "w"), indent=1)
        dll = (lambda n: os.path.join(self.lib, f"{n}.dll")) if self.windows else (lambda n: os.path.join(self.lib, f"lib{n}.so"))
        json.dump({"backend_extensions": {"shared_library_path": dll("QnnHtpNetRunExtensions"), "config_file_path": cfg}},
                  open(ext, "w"), indent=1)
        exe = os.path.join(self.bin, "qnn-context-binary-generator" + (".exe" if self.windows else ""))
        self._run([exe, "--backend", dll("QnnHtp"), "--model", dll("QnnModelDlc"), "--dlc_path", ",".join(dlc_paths),
                   "--config_file", ext, "--output_dir", out_dir, "--binary_file", name],
                  os.path.join(out_dir, name + ".ctx.log"))
        return os.path.join(out_dir, name + ".bin")

    def _context_binary_wsl(self, dlc_paths, graph_names, soc, out_dir, name):
        """То же через wsl.exe и Linux-версию QAIRT: пути Windows -> /mnt/<диск>/…"""
        soc_id, arch, vtcm = SOCS[soc]
        q = self.wsl_qairt.rstrip("/")
        if q.startswith("~"):  # путь попадает и в JSON, где ~ не раскрывается
            q = subprocess.run(["wsl.exe", "-e", "bash", "-lc", f"echo {q}"], capture_output=True, text=True).stdout.strip()
        lib = f"{q}/lib/x86_64-linux-clang"
        cfg = os.path.join(out_dir, name + ".htp.json")
        ext = os.path.join(out_dir, name + ".ext.json")
        json.dump({"graphs": [{"vtcm_mb": vtcm, "O": 3, "graph_names": graph_names}],
                   "devices": [{"device_id": 0, "soc_id": soc_id, "dsp_arch": arch,
                                "cores": [{"core_id": 0, "perf_profile": "burst", "rpc_control_latency": 100}]}],
                   "context": {"weight_sharing_enabled": len(graph_names) > 1}}, open(cfg, "w"), indent=1)
        json.dump({"backend_extensions": {"shared_library_path": f"{lib}/libQnnHtpNetRunExtensions.so",
                                          "config_file_path": self.wsl_path(cfg)}}, open(ext, "w"), indent=1)
        cmd = (f"{q}/bin/x86_64-linux-clang/qnn-context-binary-generator --backend {lib}/libQnnHtp.so "
               f"--model {lib}/libQnnModelDlc.so --dlc_path {','.join(self.wsl_path(p) for p in dlc_paths)} "
               f"--config_file {self.wsl_path(ext)} --output_dir {self.wsl_path(out_dir)} --binary_file {name}")
        env = f". {self.wsl_env} >/dev/null 2>&1; " if self.wsl_env else ""
        script = f"{env}export LD_LIBRARY_PATH={lib}:$LD_LIBRARY_PATH; {cmd}"
        with open(os.path.join(out_dir, name + ".ctx.log"), "w") as log:
            r = subprocess.run(["wsl.exe", "-e", "bash", "-lc", script], stdout=log, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            tail = open(os.path.join(out_dir, name + ".ctx.log"), errors="replace").read()[-3000:]
            raise RuntimeError(f"qnn-context-binary-generator (WSL) завершился с кодом {r.returncode}:\n{tail}")
        return os.path.join(out_dir, name + ".bin")
