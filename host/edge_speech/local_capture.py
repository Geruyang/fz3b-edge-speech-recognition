"""Normalize Windows PCM and keep a local WAV while the board reads it."""
import audioop
from array import array
from collections import deque
import math
from threading import Condition
import wave

from .layer_stream import SAMPLE_RATE

ADAPTIVE_PAUSE_FRAMES = 10
ADAPTIVE_PREROLL_FRAMES = 8


class Pcm16:
    def __init__(self, rate, channels, kind):
        self.rate = rate
        self.channels = channels
        self.kind = kind
        self.width = {"int16": 2, "float": 4, "int32": 4, "int24": 3, "uint8": 1}[kind]
        self.frame_width = self.width * channels
        self.tail = b""
        self.state = None

    def push(self, data):
        data = self.tail + data
        limit = len(data) - len(data) % self.frame_width
        self.tail = data[limit:]
        data = data[:limit]
        if not data:
            return b""
        if self.kind == "float":
            values = array("f")
            values.frombytes(data)
            data = array("h", (
                max(-32768, min(32767, round(v * 32768))) if math.isfinite(v) else 0
                for v in values
            )).tobytes()
        elif self.kind == "uint8":
            data = audioop.lin2lin(audioop.bias(data, 1, -128), 1, 2)
        elif self.width != 2:
            data = audioop.lin2lin(data, self.width, 2)
        if self.channels == 2:
            data = audioop.tomono(data, 2, 0.5, 0.5)
        elif self.channels > 2:
            values = array("h")
            values.frombytes(data)
            data = array("h", (
                sum(values[i:i+self.channels]) // self.channels
                for i in range(0, len(values), self.channels)
            )).tobytes()
        if self.rate != SAMPLE_RATE:
            data, self.state = audioop.ratecv(data, 2, 1, self.rate, SAMPLE_RATE,
                                              self.state, 3, 2)
        return data


class WavSpool:
    """Use a disk-backed queue so capture can continue during a slow connection."""
    def __init__(self, path):
        self.path = path
        self.changed = Condition()
        self.size = 0
        self.ended = False
        self.file = path.open("x+b", buffering=0)
        self.writer = wave.open(self.file, "wb")
        self.writer.setparams((1, 2, SAMPLE_RATE, 0, "NONE", ""))
        self.writer.writeframesraw(b"")

    def push(self, data):
        with self.changed:
            if data:
                self.writer.writeframesraw(data)
                self.size += len(data)
                self.changed.notify()

    def finish(self):
        with self.changed:
            if self.ended:
                return
            try:
                self.writer.close()
            finally:
                self.file.close()
                self.ended = True
                self.changed.notify_all()

    def blocks(self):
        sent = 0
        with self.path.open("rb", buffering=0) as f:
            f.seek(44)
            while True:
                with self.changed:
                    self.changed.wait_for(lambda: self.size > sent or self.ended)
                    count = min(self.size - sent, 8192)
                if not count:
                    return
                data = f.read(count)
                if len(data) != count:
                    raise OSError("录音文件读取不完整")
                sent += count
                yield data


def wav_blocks(path):
    with wave.open(str(path), "rb") as wav:
        kind = {1: "uint8", 2: "int16", 3: "int24", 4: "int32"}.get(wav.getsampwidth())
        if kind is None:
            raise ValueError("请选择 PCM WAV 文件")
        pcm = Pcm16(wav.getframerate(), wav.getnchannels(), kind)
        while block := wav.readframes(4096):
            if data := pcm.push(block):
                yield data


def pcm_frames(blocks):
    """Split PCM16 into 20 ms frames, preserving a short final frame."""
    width = 640
    buf = bytearray()
    for block in blocks:
        buf.extend(block)
        end = len(buf) - len(buf) % width
        for offset in range(0, end, width):
            yield bytes(buf[offset:offset+width])
        del buf[:end]
    if buf:
        yield bytes(buf)


def pause_segments(blocks, quiet_rms=100, pause_frames=42):
    """Expose contiguous PCM ranges after confirmed pauses, preserving all bytes."""
    stream = pcm_frames(blocks)

    def until_pause(first):
        sounded = False
        silent = 0
        data = first
        while data is not None:
            yield data
            if audioop.rms(data, 2) > quiet_rms:
                sounded = True
                silent = 0
            elif sounded:
                silent += 1
                if silent >= pause_frames:
                    return
            data = next(stream, None)

    while (first := next(stream, None)) is not None:
        yield until_pause(first)


def adaptive_segments(blocks, pause_frames=ADAPTIVE_PAUSE_FRAMES, preroll_frames=ADAPTIVE_PREROLL_FRAMES):
    """Reset at speech resumption after a pause, preserving onset pre-roll."""
    stream = pcm_frames(blocks)
    carry = deque()
    floor = 72.0

    def until_resumption(first):
        nonlocal floor
        quiet_frames = 0
        voice_run = 1 if carry else 0
        voiced_frames = voice_run
        armed = False
        pre_roll = deque()
        while carry:
            # These frames were already classified by the previous segment.
            yield carry.popleft()
        data = first
        while data is not None:
            level = audioop.rms(data, 2)
            threshold = max(72, min(420, floor*2.4+32))
            if level <= threshold:
                floor += (level-floor)*0.06
                voice_run = 0
                if armed:
                    quiet_frames += 1
                    pre_roll.append(data)
                    if len(pre_roll)>preroll_frames:
                        yield pre_roll.popleft()
                else:
                    yield data
            else:
                if armed and quiet_frames>=pause_frames:
                    carry.extend(pre_roll)
                    carry.append(data)
                    return
                yield from pre_roll
                pre_roll.clear()
                quiet_frames = 0
                voice_run += 1
                voiced_frames += 1
                armed = armed or (voice_run>=4 and voiced_frames>=10)
                yield data
            data = next(stream, None)
        yield from pre_roll

    first = next(stream, None)
    while first is not None or carry:
        yield until_resumption(first)
        first = next(stream, None)
