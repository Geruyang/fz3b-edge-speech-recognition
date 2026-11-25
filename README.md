# FZ3B Edge Speech Recognition V1.0

Windows 从 USB 摄像头内置麦克风采集语音，通过直连网线传给 FZ3B。板端以 Zipformer 进行中文、英文和中英混合转写，再把部分与最终原稿送回客户端。

| 位置 | V1.0 实现 |
| --- | --- |
| PS | Cortex-A53 / ARM64。运行流式服务、声学特征、INT8 Embedding、层间调整、末端投影、Decoder 与 Joiner |
| PL | 同频 200 MHz 的 INT8 编码层核，分批复用十个编码层，INT32 累加，通过 HP0 访问 DDR |
| Windows | USB 采集、实时原稿、三种分段模式、WAV 和 TXT 保存。本地 Qwen 可另存整理稿 |

## 下载

| Release 文件 | 内容 |
| --- | --- |
| FZ3B-Board-V1.0.tar.gz | ARM64 应用、内核模块、设备树、PCAP、PS 模型和 PL 权重、初始化和首组核验 |
| FZ3B-PL-V1.0.zip | 与实板相同配置的 BIT、XSA 和资源、布线、时序报告 |
| FZ3B-Zipformer-INT8-Weights-V1.0.zip | 单独下载的 Zipformer PS ONNX 图、词表、层表、PL 权重和参数 |
| FZ3B-Windows-V1.0.zip | Windows 便携客户端，内部为已实测的客户端 1.3，内含本地 Qwen2.5-0.5B-Instruct Q4_K_M 权重 |
| SHA256SUMS.txt | 各发布包的 SHA-256 |

模型与发布工件身份见 [manifests](manifests/)。模型来源和第三方声明见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 使用已验证的版本

板端发布包面向使用 Linux 4.14.0-xilinx-v2018.3 的 FZ3B。语音驱动按原板的设备树、DMA 范围和时钟核验。handoff.py 依赖已验证的 A0005 服务及其配套加载器，负责移交 PL。安装或启动前，请比对现有系统的设备、服务与恢复条件。板载系统和恢复镜像由设备所有者另行保管。

发布包根目录为 fz3b-int8-desktop-board。受控部署时，将其放到板端 /root/workspace/edgespeech-int8/ 下的版本目录，核对 manifest.json 和 python3 -B activate.py --verify 后，将同级 current 指向该版本。正常启动后，Windows 首次连接会执行匹配的初始化并通过数值核验。板端使用现有 SSH 账户。

解压 Windows 包后双击 FZ3B-Transcribe.cmd。发布客户端的缺省 SSH 主机指纹用于本次实测板卡。使用另一块板时，先从其本地串口或管理渠道确认真实指纹，再在 host/edge_speech/layer_stream.py 设置 BOARD_PIN 并重新打包。[客户端使用说明](host/edge_speech/desktop_help.md)

## 源码与构建入口

- PS 应用入口在 hardware/candidates/int8_layer/board/sentence_live.cpp。ARM64 CMake 构建需与目标 Linux 兼容的 sysroot、ONNX Runtime 1.17.1 头文件、nlohmann/json 头文件及 kaldi-native-fbank 1.20.0 源码。启用 EDGE_SPEECH_ENABLE_ORT_PL 和 EDGE_SPEECH_ENABLE_ASR_JOBS，将 BUILD_TESTING 设为 OFF，构建 desktop-serve。
- PS 内核模块在 hardware/candidates/int8_layer/board/driver，用与 FZ3B 运行内核匹配的 ARM64 工具链和构建树编译。该目录包含模块依赖的冻结哈希、头文件及设备树源。
- PL Vitis HLS 2026.1 源在 hardware/candidates/int8_layer/kernel，综合配置为 request.cfg。Vivado 平台输入、D4 时钟源和 PS 预设随源码提交。依次运行 rtl/platform.tcl、rtl/implement.tcl、rtl/export_routed.tcl。platform.tcl 参数依次为已生成的 HLS IP 目录、输出目录和 96 位硬件标识。已验证位流标识为 0x51E36A2FAFB04A9B68924206。
- Windows 源码在 host/edge_speech。Python 3.12 环境可用 python -m host.edge_speech.desktop 启动。tools/package_desktop.py 支持已有 PySide6、SSH 依赖及可选本地 Qwen 资源目录。

九段软件合成语音通过扬声器和摄像头麦克风配对实测。智能分段把英文漏词从 44/69 降到 2/69，整段遗漏从 7/11 降到 0/11。同一录音的首稿与收尾等待配对中位分别增加 15 和 3 毫秒。该结果来自带停顿的固定测试。
