# ORT 流式 ASR 后端

CPU 与 PL 模式都通过 `ChunkBackend` 交给 `StreamEngine`。PL 模式向设备提交当前 `39×80` F32 特征，把通过身份、完成和排空检查的 `16×256` F32 回读传入下游 encoder。下游在 PS 执行 50 个冻结 W8 FFN/Conv 块以及原模型的 Attention、残差和 BasicNorm。decoder、joiner 和词表沿用原发布工件。

`make_ort_pl_backend(paths, submit)` 的回调接收 `ChunkStamp` 及实时 `FeatureWindow`。返回的 `OrtEmbeddingResult` 包含相同 stamp 和按时间优先排列的特征。设备适配持有文件描述符、物理作业身份和排空责任。下游抛错时，原 encoder 缓存、decoder context 和输出 token 保持可用。相同 stamp 与相同输入重试复用最近一次已成功完成的设备结果。设备错误、过期或非有限回执使该后端停止提交。模型硬边界由 `StreamEngine` 分配新的流代次，设备物理序号继续由持有它的适配器管理。

## 从 WAV 或麦克风启动板端应用

`edge_speech_board` 默认沿用 CPU 识别。选择 `--asr-backend pl` 时，WAV 和 ALSA 采集均经过 kaldi-native-fbank 1.20.0，再由设备计算 Embedding。服务在识别器就绪后启动录音。固定录音从 `--wav` 指定，麦克风采集可指定 `--alsa-rate 16000`。同一会话按顺序提交，FZAS 的 `stop` 包含末块解码，`complete` 释放会话。真实设备模式要求维护方预先建立就绪窗口。

在仓库根目录可使用同一个服务入口回放之前的实板结果。每次提交都逐字节核对当前 WAV 算出的输入，匹配后读取对应的已封存输出。可选的 `--asr-trace` 将设备回执、结束时的 token 和已完成解码步数写入独立 JSONL 文件。

```powershell
build/t018-app-host/edge_speech_board.exe --asr-backend pl --pl-encoder build/t019-ort-pl/encoder-pl.onnx `
    --model-dir assets/reference-models --ort-library build/t008-ort-windows/onnxruntime.dll `
    --wav assets/reference-models/0.wav --embedding-capture reports/t018/stream-board-live/window/export `
    --journal build/t018-app-host/transcripts.jsonl --asr-trace build/t018-app-host/trace.jsonl
```

## 构建运行时

PL 应用的两个发布目标使用同一份 KNF 源码。原始压缩包、下载地址和 SHA-256 在 `toolchains/versions.lock.json` 的 `pl_fbank` 中。下面的命令使用已核验并解包的源码和项目锁定的 ORT、JSON 头文件，在 Windows 工程根目录执行。

```powershell
$headers = 'build/t002-ort-aarch64/archive/onnxruntime-linux-aarch64-glibc2_17-Release-1.17.1-patched/include'
$flags = @('-DCMAKE_BUILD_TYPE=Release', '-DEDGE_SPEECH_ENABLE_ORT_PL=ON',
    '-DEDGE_SPEECH_KNF_SOURCE_DIR=build/t018-arm-deps/kaldi-native-fbank-1.20.0',
    "-DEDGE_SPEECH_ORT_INCLUDE_DIR=$headers", '-DEDGE_SPEECH_JSON_INCLUDE_DIR=build/phase3-json')
cmake -S . -B build/t018-app-host -G Ninja @flags
cmake --build build/t018-app-host --target edge_speech_board
python tools/prepare_bionic_arm64_sysroot.py --offline
cmake -S . -B build/t018-arm-app -G Ninja @flags '-DCMAKE_TOOLCHAIN_FILE=tools/cmake/t018-aarch64-bionic.cmake' -DBUILD_TESTING=OFF
cmake --build build/t018-arm-app --target edge_speech_board
```

首次准备 Bionic 编译环境时省略 `--offline`，脚本按 SHA-256 下载九个所需软件包。ARM64 产物静态链接 KNF，动态装载板端私有 ORT 1.17.1。Windows 使用锁定的 ORT 1.18.1。两个平台的数学运行库会影响特征的最低有效位，严格回放结果按实际输入分别记录。

## 导出 PL 模型

在仓库根使用现有冻结模型工具环境。脚本读取锁定的 FP32 encoder，从其中生成 PS 量化权重，并写入一个 ONNX 文件。板端加载这个文件和原 decoder、joiner、词表，不需要历史输入和输出夹具。

```powershell
$env:PYTHONPATH="$PWD/build/model-inspect-deps;$PWD/build/t005-fbank-deps"
python -m model_tools.export_ort_pl_encoder build/t019-ort-pl/encoder-pl.onnx
```

发布模型 ID 为 `zipformer-bilingual-e2382758-w8-pl-embedding`。导出 ONNX 的 SHA-256 为 `2e357e72f9dc14c5ae4fce57d5358f1ec059a5c669c21920ada5b867013a1e27`，输入 `pl_embedding` 是静态 `[1,16,256]` F32。模型另保留 35 个 encoder 状态输入与输出。块参数使用 ONNX 内嵌的 `raw_data`。

## 构建与后端数值测试

PL 后端使用板端私有 ORT 1.17.1 头文件和运行库。Windows 主机可用同一套 1.17.1 头文件及现有 ORT 1.18.1 DLL。编译 `board/asr/ort_backend.cpp` 和 `board/asr/ort_pl_custom.cpp` 时启用 `EDGE_SPEECH_ENABLE_ORT_PL=1`。链接 `hardware/hls/asr_blocks_shared_quant.cpp`、`hardware/hls/vector_ops.cpp`、`hardware/hls/mac_array.cpp` 与线程库。用 GNU 编译器时关闭 FP contraction 以保留量化与反量化的舍入点。

以下本机命令用已归档的实板数据验证 C++ 后端的设备回调和解码。真实特征逐字节校对、最终 token、过期回执与已完成作业的失败重试均在测试中检查。第二项负控会打印预期的 ORT 算子错误。

```powershell
$cc=(Get-Command g++.exe).Path
$env:PATH=(Split-Path $cc)+';'+$env:PATH
$ortHeaders=(Resolve-Path build/t002-ort-aarch64/archive/onnxruntime-linux-aarch64-glibc2_17-Release-1.17.1-patched/include).Path
& $cc -std=c++17 -O2 -ffp-contract=off -static -DEDGE_SPEECH_ENABLE_ORT_PL=1 `
  -I . -I $ortHeaders tests/integration/ort_pl_backend_test.cpp board/asr/ort_backend.cpp `
  board/asr/ort_pl_custom.cpp hardware/hls/asr_blocks_shared_quant.cpp hardware/hls/vector_ops.cpp `
  hardware/hls/mac_array.cpp -o build/t019-ort-pl/ort_pl_backend_test.exe
& build/t019-ort-pl/ort_pl_backend_test.exe build/t008-ort-windows/onnxruntime.dll `
  build/t019-ort-pl/encoder-pl.onnx assets/reference-models reports/t015/embedding-stream/fixtures `
  reports/t018/stream-board-live/window/export
```

## AArch64 交叉构建

在仓库根运行。`prepare_bionic_arm64_sysroot` 校验 Ubuntu 18.04 ARM64 包的 SHA-256，分别提供 Glibc 2.27、GCC 8.4 C++ 库及 ALSA。已有包缓存时加 `--offline`。交叉编译使用 AMD GCC 13.4，目标头文件、启动对象和共享库来自 Bionic sysroot。ORT 在运行时从应用指定的私有 1.17.1 库加载。前端源目录指向解包后的 kaldi-native-fbank 1.20.0。

```powershell
python -m tools.prepare_bionic_arm64_sysroot --offline
cmake -S . -B build/t018-arm-app -G Ninja `
  -DCMAKE_TOOLCHAIN_FILE=tools/cmake/t018-aarch64-bionic.cmake `
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DEDGE_SPEECH_ENABLE_ORT_PL=ON `
  -DEDGE_SPEECH_ORT_INCLUDE_DIR=build/t002-ort-aarch64/archive/onnxruntime-linux-aarch64-glibc2_17-Release-1.17.1-patched/include `
  -DEDGE_SPEECH_JSON_INCLUDE_DIR=build/phase3-json `
  -DEDGE_SPEECH_KNF_SOURCE_DIR=build/t018-arm-deps/kaldi-native-fbank-1.20.0
cmake --build build/t018-arm-app --target edge_speech_board
```

`python tools/t018_qemu_board_app.py` 用工厂内核和上述 Bionic 运行库装载应用，在 RAM guest 内通过本地 FZAS 会话测试 `start`、`status`、`stop`、`resume`。默认重放已归档的实板原始回读。若首窗实际特征与捕获输入不一致，测试要求错误会话没有完成回执、转录、冻结或快照。
