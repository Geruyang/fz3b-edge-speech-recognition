import struct
from threading import Thread
import wave

from host.edge_speech.local_capture import Pcm16, WavSpool, adaptive_segments, pause_segments
from host.edge_speech.local_qwen import accepted, pieces


def test_recording_keeps_odd_tail_and_order_when_consumer_waits(tmp_path):
    spool = WavSpool(tmp_path/"test.wav")
    pcm = struct.pack("<1513h", *range(1513))
    received = []
    task = Thread(target=lambda: received.extend(spool.blocks()))
    task.start()
    spool.push(pcm[:640])
    spool.push(pcm[640:])
    spool.finish()
    task.join(3)
    assert not task.is_alive()
    with wave.open(str(spool.path), "rb") as f:
        assert f.getnframes() == 1513
        assert f.readframes(1513) == pcm == b"".join(received)


def test_fragmented_stereo_conversion_preserves_resampling_phase():
    data = struct.pack("<1600h", *range(1600))
    complete = Pcm16(48000, 2, "int16").push(data)
    streaming = Pcm16(48000, 2, "int16")
    assert b"".join(streaming.push(data[i:i+153]) for i in range(0, len(data), 153)) == complete


def test_qwen_preserves_numbers_negation_and_original_coverage():
    raw = "今天没有重启 FZ3B 原稿分为 19 个 token"
    assert accepted(raw, "今天没有重启 FZ3B，原稿分为 19 个 token。")
    assert not accepted(raw, "今天重启 FZ3B，原稿分为 19 个 token。")
    assert not accepted(raw, raw.replace("19", "20"))
    assert not accepted(raw, "今天没有重启 FZ3B，原稿分为 19 个 token。谢谢大家。")
    long = (raw+"。")*17
    assert "".join(pieces(long)) == long


def test_pause_segmentation_keeps_all_samples_and_creates_new_context_after_quiet():
    voice = struct.pack("<h", 2000)*320
    quiet = bytes(640)
    phrase = voice*4 + quiet*42
    second = voice*3 + b"\x14\x03"
    pcm = phrase + second
    fragments = [pcm[i:i+1234] for i in range(0, len(pcm), 1234)]
    assert [b"".join(part) for part in pause_segments(fragments)] == [phrase, second]


def test_adaptive_pause_waits_for_resumption_and_preserves_preroll_and_tail():
    voice = struct.pack("<h", 2000)*320
    noise = struct.pack("<h", 115)*320
    frames = [noise]*30+[voice]*12+[noise]*16+[voice]*10+[noise]*50
    tail = b"\x65\x00"
    pcm = b"".join(frames)+tail
    chunks = (pcm[i:i+818] for i in range(0,len(pcm),818))
    segments = [b"".join(s) for s in adaptive_segments(chunks)]
    assert segments == [b"".join(frames[:50]), b"".join(frames[50:])+tail]
