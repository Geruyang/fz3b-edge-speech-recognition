"""Restore the frozen FZ3B speech application after a normal factory boot."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


RELEASE = Path(__file__).resolve().parent
MODULE = "fz_layer_compute"
DEVICE = Path("/dev/fz-layer-live")
DT = Path("/sys/kernel/config/device-tree/overlays")
FIRMWARE = Path("/lib/firmware/fzspeech/int8-layer-request")
LOCKS = ("/run/lock/fz3b-pl-loader.lock", "/run/fz3b-r5-start.lock", "/run/fz3b-vision-r5-owner.lock")


def check(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def verify():
    manifest = json.loads((RELEASE/"manifest.json").read_text())
    for name, digest in manifest["files"].items():
        p = RELEASE/name
        check(not p.is_symlink() and p.resolve() == p and hashlib.sha256(p.read_bytes()).hexdigest() == digest, name)
    return manifest


def guard(directory):
    program = (
        "import fcntl,os,sys,time\nfrom pathlib import Path\n"
        "fds=[open(p,'a') for p in sys.argv[2:]]\n"
        "for f in fds: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
        "d=Path(sys.argv[1]); (d/'lock-pid').write_text(str(os.getpid()))\n"
        "while not (d/'release-locks').exists(): time.sleep(.2)\n"
    )
    with (directory/"lock.log").open("xb") as log:
        process = subprocess.Popen(
            [sys.executable, "-Bu", "-c", program, str(directory)] + list(LOCKS),
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True
        )
    for _ in range(100):
        p = directory/"lock-pid"
        if p.is_file() and p.read_text() == str(process.pid):
            return
        check(process.poll() is None, "原服务仍持有板卡资源")
        time.sleep(.05)
    raise RuntimeError("未能取得板卡维护租约")


def activate():
    verify()
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    directory = Path("/run")/("fz-int8-"+boot.split("-")[0])
    with open("/run/edgespeech-activate.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ready = directory/"desktop-current"
        if DEVICE.is_char_device() and (ready/"desktop-serve").is_file():
            return
        check(not os.path.lexists(str(directory)), "当前启动已存在维护现场")
        check(not DEVICE.exists() and not (Path("/sys/module")/MODULE).exists(), "已有层核仍在运行")
        check(not (DT/"fz-int8-layer-request").exists(), "已有语音设备树")
        if os.path.lexists(str(FIRMWARE)):
            check(FIRMWARE.is_symlink() and FIRMWARE.resolve() == RELEASE/"firmware", "已有不同固件路径")
        directory.mkdir(mode=0o700)
        with (directory/"activation.log").open("x") as log:
            # Use the same checked handoff that was exercised before the first layer job.
            source = (RELEASE/"handoff.py").read_text().replace("BUILD_BOOT_UUID", boot)
            subprocess.run([sys.executable, "-B", "-"], input=source.encode(),
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=100)
        guard(directory)
        FIRMWARE.parent.mkdir(exist_ok=True)
        if not os.path.lexists(str(FIRMWARE)):
            FIRMWARE.symlink_to(RELEASE/"firmware")
        overlay = DT/"fz-int8-layer-request"
        overlay.mkdir()
        (overlay/"dtbo").write_bytes((RELEASE/"layer.dtbo").read_bytes())
        check((overlay/"status").read_text().strip() == "applied", "语音设备树未生效")
        base = Path("/sys/firmware/devicetree/base")
        conflicts = ("amba_pl@0/mipi_axi@a0010000", "amba_pl@0/reg_irq_path@a0000000",
                     "amba_pl@0/serial@a0020000", "fpga-driver@0")
        for name in conflicts:
            check((base/name/"status").read_bytes().rstrip(b"\0") == b"disabled", name)
        subprocess.check_call(["/sbin/insmod", str(RELEASE/(MODULE+".ko"))])
        for _ in range(250):
            if DEVICE.is_char_device():
                break
            time.sleep(.1)
        check(DEVICE.is_char_device(), "语音驱动未就绪，请查看板端内核日志")
        local = directory/"smoke"
        local.mkdir(mode=0o700)
        for name in ("input", "old", "output", "next"):
            (local/(name+".bin")).symlink_to(RELEASE/"smoke"/(name+".bin"))
        smoke = "from pathlib import Path\nWD=Path(" + repr(str(directory)) + ")\n" + (RELEASE/"smoke.py").read_text()
        with (directory/"smoke.log").open("x") as log:
            subprocess.run([sys.executable, "-B", "-"], input=smoke.encode(),
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=8)
        check(FIRMWARE.is_symlink() and FIRMWARE.resolve() == RELEASE/"firmware", "语音固件链接已改变")
        FIRMWARE.unlink()
        ready.symlink_to(RELEASE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(), sort_keys=True))
    else:
        activate()
