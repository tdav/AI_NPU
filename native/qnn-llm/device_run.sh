#!/usr/bin/env bash
# Проверка пакета qnn-llm-convert на телефоне через adb (WSL):
#   device_run.sh <каталог пакета> "<запрос>" [max_new] [temperature]
# Пакет копируется в /data/local/tmp/qnn-llm/<имя> (уже скопированные файлы того же размера пропускаются),
# промпт собирается по metadata.json ("chat"), ответ декодируется tokenizer.json. Нужен python с tokenizers.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
pkg=$(realpath "$1"); prompt=$2; max_new=${3:-128}; temp=${4:-0}
ADB=${ADB:-"/mnt/c/Program Files (x86)/Android/android-sdk/platform-tools/adb.exe"}
PY=${PY:-python3}
name=$(basename "$pkg"); [ "$name" = package ] && name=$(basename "$(dirname "$pkg")")  # work/<модель>/package
dev=/data/local/tmp/qnn-llm; ddir=$dev/$name
libs="$here/../../AiNpu/Platforms/Android/qnn/arm64-v8a"
"$ADB" shell mkdir -p "$ddir"
push() {  # push <локальный файл> <каталог на устройстве>
  local f=$1 d=$2 size remote
  size=$(stat -c%s "$f"); remote=$("$ADB" shell "stat -c%s $d/$(basename "$f") 2>/dev/null" | tr -d '\r' || true)
  [ "$remote" = "$size" ] || "$ADB" push "$(wslpath -w "$f")" "$d/" > /dev/null
}
for f in "$libs"/*.so "$here/build/qnn-llm-cli"; do push "$f" $dev; done
for f in "$pkg"/*; do push "$f" "$ddir"; done
"$ADB" shell chmod +x $dev/qnn-llm-cli
read -r ids eos < <("$PY" - "$pkg" "$prompt" <<'EOF'
import json, sys
from tokenizers import Tokenizer
pkg, text = sys.argv[1], sys.argv[2]
m = json.load(open(f"{pkg}/metadata.json")); c = m["chat"]
t = Tokenizer.from_file(f"{pkg}/tokenizer.json")
ids = t.encode(c["bos"] + c["user"].replace("{content}", text) + c["assistant"], add_special_tokens=False).ids
print(",".join(map(str, ids)), ",".join(map(str, m["eos"])))
EOF
)
out=$("$ADB" shell "cd $dev && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./qnn-llm-cli $dev $ddir $ids $max_new $temp $eos" 2>&1 | tr -d '\r')
echo "$out" | grep -E "init|STAT|ERROR|perf"
echo "$out" | "$PY" -c '
import sys
from tokenizers import Tokenizer
t = Tokenizer.from_file(sys.argv[1] + "/tokenizer.json")
line = [l for l in sys.stdin.read().splitlines() if l.startswith("IDS:")]
print("ОТВЕТ:", t.decode([int(v) for v in line[0][4:].split(",") if v]) if line else "—")
' "$pkg"
