"""Build a relocatable, offline Windows client using the pinned local dependencies."""
import argparse
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "host/edge_speech"


def copy(source, dest):
    if source.is_dir():
        shutil.copytree(source, dest, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)


def package(output, qt, ssh, qwen=None):
    output.mkdir(parents=True, exist_ok=False)
    runtime = output / "runtime"
    python = Path(sys.base_prefix)
    for name in ("pythonw.exe", "python.exe", "python3.dll", "python312.dll",
                 "vcruntime140.dll", "vcruntime140_1.dll", "LICENSE.txt", "DLLs"):
        copy(python / name, runtime / name)
    shutil.copytree(python / "Lib", runtime / "Lib", ignore=shutil.ignore_patterns(
        "site-packages", "__pycache__", "*.pyc", "test", "tests", "idlelib", "turtledemo", "tkinter"
    ))
    deps = runtime / "Lib/site-packages"
    for p in ssh.iterdir():
        if p.name != "bin":
            copy(p, deps / p.name)
    for p in qt.iterdir():
        if p.name == "shiboken6" or p.name.endswith(".dist-info"):
            copy(p, deps / p.name)
    pyside = qt / "PySide6"
    modules = ("QtCore", "QtGui", "QtWidgets", "QtMultimedia", "QtNetwork")
    selected = {"__init__.py", "_config.py", "_git_pyside_version.py", "pyside6.abi3.dll", "qt.conf"}
    selected.update(m+".pyd" for m in modules)
    selected.update("Qt6"+m[2:]+".dll" for m in modules)
    selected.update({"Qt6OpenGL.dll", "d3dcompiler_47.dll", "opengl32sw.dll",
                     "avcodec-61.dll", "avformat-61.dll", "avutil-59.dll", "swresample-5.dll", "swscale-8.dll"})
    for p in pyside.iterdir():
        if p.name in selected or p.name.startswith(("msvcp140", "vcruntime140")):
            copy(p, deps / "PySide6" / p.name)
    for name in ("platforms", "multimedia", "styles"):
        copy(pyside / "plugins" / name, deps / "PySide6/plugins" / name)
    for name in ("desktop.py", "local_capture.py", "layer_stream.py", "local_qwen.py", "client.py", "types.py"):
        copy(APP / name, output / "host/edge_speech" / name)
    copy(ROOT / "host/edge_speech/desktop_help.md", output / "README.md")
    if qwen:
        copy(qwen,output/"qwen")
    # Keep Python isolated from globally installed packages and PATH.
    (runtime / "python312._pth").write_text(".\nDLLs\nLib\nLib/site-packages\n..\n", encoding="ascii")
    (output / "launch.pyw").write_text("from host.edge_speech.desktop import main\nraise SystemExit(main())\n", encoding="ascii")
    (output / "FZ3B-Transcribe.cmd").write_text(
        '@echo off\nstart "" "%~dp0runtime\\pythonw.exe" "%~dp0launch.pyw"\n', encoding="ascii"
    )
    archive = output.parent / (output.name + ".zip")
    files = sorted(p for p in output.rglob("*") if p.is_file())
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for file in files:
            z.write(file, output.name + "/" + file.relative_to(output).as_posix())
    print(archive, archive.stat().st_size)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    cli.add_argument("--qt", type=Path, default=ROOT/"build/phase3-qt-deps")
    cli.add_argument("--ssh", type=Path, required=True)
    cli.add_argument("--qwen", type=Path)
    config = cli.parse_args()
    package(config.output.resolve(), config.qt, config.ssh, config.qwen)
