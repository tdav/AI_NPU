#!/usr/bin/env bash
# Сборка libqnn-llm.so и qnn-llm-cli (Android arm64) в WSL.
# Требует: NDK r29 в ~/android/android-ndk-r29, заголовки QNN 2.40 (QAIRT 2.40.0.251030) в $QNN_INCLUDE.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
NDK=${ANDROID_NDK:-~/android/android-ndk-r29}
QNN_INCLUDE=${QNN_INCLUDE:-~/qairt/qairt/2.40.0.251030/include/QNN}
CXX=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang++
out=${1:-$here/../../AiNpu/Platforms/Android/qnn/arm64-v8a}
mkdir -p "$out" "$here/build"
# libc++ статически: в APK не нужен libc++_shared.so; max-page-size 16 КБ — требование Android 15+.
$CXX -O2 -std=c++17 -fPIC -shared -I "$QNN_INCLUDE" -I "$here" "$here/qnn_llm.cc" -o "$out/libqnn-llm.so" \
  -static-libstdc++ -ldl -llog -Wl,--version-script="$here/qnn-llm.lds" -Wl,-z,max-page-size=16384
$CXX -O2 -std=c++17 "$here/qnn_llm_cli.cc" -o "$here/build/qnn-llm-cli" -L "$out" -lqnn-llm -Wl,-rpath,'$ORIGIN'
ls -l "$out/libqnn-llm.so" "$here/build/qnn-llm-cli"
