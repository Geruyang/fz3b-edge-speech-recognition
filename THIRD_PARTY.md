# Third-party components

The Zipformer model builds on the [bilingual Chinese-English Zipformer model](https://huggingface.co/csukuangfj/k2fsa-zipformer-bilingual-zh-en-t). The deployed PS ONNX parts and the PL packed weights in V1.0 are quantized and transformed derivatives. Upstream lists Apache-2.0. The Apache license text accompanies the model asset.

The Windows bundle includes [Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF), Apache-2.0, and the [llama.cpp b11865 CPU runtime](https://github.com/ggml-org/llama.cpp/releases/tag/b11865), MIT. Its exact model checksum is in manifests/qwen.json.

The board bundle includes the ARM64 build of [ONNX Runtime 1.17.1](https://github.com/microsoft/onnxruntime/tree/v1.17.1), MIT, and uses [kaldi-native-fbank 1.20.0](https://github.com/csukuangfj/kaldi-native-fbank). The Linux module carries a GPL declaration and its application-specific source is in the repository.

Windows dependencies include CPython, PySide6/Qt and Paramiko. PySide6 and Qt 6.8.3 are dynamically loaded and provided under LGPL-3.0. The portable bundle includes the LGPL and GPL texts. Corresponding source is available via the [PySide 6.8.3 tag](https://code.qt.io/cgit/pyside/pyside-setup.git/tag/?h=v6.8.3) and [Qt 6.8.3 source archives](https://download.qt.io/archive/qt/6.8/6.8.3/single/). The Qt libraries and plugins are separate files in the portable folder and may be replaced with compatible builds. Other dependency notices accompany their distributions.

The PL project targets AMD tools and uses AMD IP blocks from the installed IP catalog. The release artifact contains the implemented hardware for the target board. AMD tool and IP terms apply. Individual source headers continue to govern their files.
