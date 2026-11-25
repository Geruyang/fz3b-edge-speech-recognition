"""Host-to-FZ3B PCM stream and transcript receipts over a pinned SSH session."""
from contextlib import contextmanager
from itertools import chain
import socket
import struct
from threading import Thread

from .client import _authenticate, _verify_host_key

BOARD_HOST = "169.254.220.254"
BOARD_PIN = "SHA256:eeDWKcszE4gC/aIC5galQocdjTPZv9oogg8pm82rWiw"
SAMPLE_RATE = 16000
MAX_PACKET = 8192
COMMAND = (
    "boot=$(cut -d- -f1 /proc/sys/kernel/random/boot_id) && "
    'app="/run/fz-int8-$boot/desktop-current" && '
    'if ! test -x "$app/desktop-serve"; then '
    '/usr/bin/python3 -B /root/workspace/edgespeech-int8/current/activate.py || exit; fi && '
    'cd "$app" && exec ./desktop-serve --stream'
)


@contextmanager
def connect(host, username, password, expected_host_key=BOARD_PIN):
    import paramiko
    with socket.create_connection((host, 22), timeout=8) as wire, paramiko.Transport(wire) as ssh:
        ssh.banner_timeout = ssh.auth_timeout = 8
        ssh.start_client(timeout=8)
        _verify_host_key(ssh.get_remote_server_key(), expected_host_key)
        _authenticate(ssh, paramiko, username, None, password or None)
        yield ssh


def transcribe(ssh, blocks, partial=None, ready=None):
    """Send 16 kHz PCM16 and require the board to acknowledge every received sample."""
    with ssh.open_session(timeout=8) as channel:
        channel.settimeout(145)
        channel.exec_command(COMMAND)
        output = channel.makefile("rb")
        if output.readline().rstrip() != b"READY":
            error = channel.makefile_stderr("rb").read(4096).decode(errors="replace").strip()
            raise ConnectionError(error or "FZ3B 上的 INT8 语音服务未就绪")
        channel.settimeout(10)
        state = {}

        def read():
            try:
                buffer = bytearray()
                while True:
                    try:
                        data = channel.recv(32768)
                    except socket.timeout:
                        continue
                    if not data:
                        return
                    buffer.extend(data)
                    while (end := buffer.find(b"\n")) >= 0:
                        line = bytes(buffer[:end])
                        del buffer[:end+1]
                        event, delimiter, value = line.rstrip(b"\r").partition(b"\t")
                        if not delimiter:
                            raise ValueError("板卡返回了无效的转写消息")
                        if event == b"PARTIAL":
                            if partial:
                                partial(value.decode("utf-8"))
                        elif event == b"SAMPLES":
                            state["samples"] = int(value)
                        elif event == b"FINAL":
                            state["text"] = value.decode("utf-8")
                            return
                        else:
                            raise ValueError("板卡返回了未知的转写消息")
            except Exception as exc:
                state["error"] = exc

        reader = Thread(target=read, daemon=True)
        reader.start()
        if ready:
            ready()
        samples = 0
        for block in blocks:
            if not reader.is_alive():
                raise ConnectionError("录音结束前板卡已关闭转写通道")
            if not block or len(block) % 2:
                raise ValueError("PCM 样本不完整")
            for offset in range(0, len(block), MAX_PACKET * 2):
                packet = block[offset:offset + MAX_PACKET * 2]
                count = len(packet) // 2
                channel.sendall(struct.pack("<I", count) + packet)
                samples += count
        channel.sendall(struct.pack("<I", 0))
        channel.shutdown_write()
        reader.join(timeout=60)
        if reader.is_alive():
            raise TimeoutError("板卡生成最终原稿超时")
        if "error" in state:
            raise state["error"]
        if "text" not in state or state.get("samples") != samples:
            error = channel.makefile_stderr("rb").read(4096).decode(errors="replace").strip()
            raise ConnectionError(error or "板卡未确认收到全部音频")
        return state["text"], samples


def transcribe_segments(ssh, segments, partial=None, ready=None):
    texts = []
    total = 0
    for segment in segments:
        first = next(segment, None)
        if first is None:
            continue

        def append_preview(text):
            if partial:
                partial("\n".join([*texts, text]))

        text, samples = transcribe(ssh, chain((first,), segment), append_preview, ready if not total else None)
        total += samples
        if text:
            texts.append(text)
    return "\n".join(texts), total
