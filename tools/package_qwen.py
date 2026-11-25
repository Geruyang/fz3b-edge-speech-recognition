"""Prepare the optional local Qwen assets from verified upstream files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile

from host.edge_speech.local_qwen import MODEL, MODEL_SHA

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = "llama-b11865-bin-win-cpu-x64.zip"
RUNTIME_SHA = "06be7290f2e86aec5e86bdc8afda50c211163b30c89d3e01726bfb3fc32552a4"


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def build(downloads, output):
    model = downloads/MODEL
    runnable = downloads/RUNTIME
    if digest(model) != MODEL_SHA or digest(runnable) != RUNTIME_SHA:
        raise ValueError("Qwen or llama.cpp source hash")
    binary = output/"runtime"
    binary.mkdir(parents=True, exist_ok=True)
    runtime = {}
    with ZipFile(runnable) as zip_file:
        for name in zip_file.namelist():
            if Path(name).name != name:
                continue
            if (name in ("llama-server.exe","llama-server-impl.dll","llama.dll","llama-common.dll",
                         "libomp.dll","mtmd.dll","LICENSE","LICENSE.txt")
                    or (name.startswith(("ggml", "msvcp", "vcruntime")) and name.endswith(".dll"))):
                content = zip_file.read(name)
                (binary/name).write_bytes(content)
                runtime["runtime/"+name] = hashlib.sha256(content).hexdigest()
    if "runtime/llama-server.exe" not in runtime:
        raise ValueError("llama-server missing")
    shutil.copyfile(model,output/MODEL)
    shutil.copyfile(ROOT/"assets/reference-models/qwen/LICENSE",output/"Qwen-LICENSE.txt")
    manifest = dict(model=MODEL,model_sha256=MODEL_SHA,
                    model_source="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                    llama_cpp="b11865",llama_cpp_zip_sha256=RUNTIME_SHA,
                    llama_cpp_source="https://github.com/ggml-org/llama.cpp/releases/tag/b11865",
                    runtime=runtime)
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("downloads",type=Path)
    cli.add_argument("output",type=Path)
    args=cli.parse_args()
    build(args.downloads,args.output)
