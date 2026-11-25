"""Prepare mixed utterances with exact synthetic gap durations before acoustic playback."""
import argparse
import audioop
import json
from pathlib import Path
import wave

from host.edge_speech.local_capture import wav_blocks


def trim(data):
    # Keep a small margin before and after speech to protect quiet consonants.
    width = 160 * 2
    active = [i for i in range(0, len(data), width) if audioop.rms(data[i:i+width], 2) >= 160]
    if not active:
        raise ValueError("synthesized source has no speech")
    return data[max(0, active[0]-width*3):active[-1]+width*4]


def generate(spec, sources, output):
    data = json.loads(spec.read_bytes())
    output.mkdir(parents=True, exist_ok=False)
    clips = {}
    for entry in data["utterances"]:
        raw = trim(b"".join(wav_blocks(sources/(entry["name"]+".wav"))))
        gain = min(3900/audioop.rms(raw,2), 27000/audioop.max(raw,2))
        clips[entry["name"]] = (entry, audioop.mul(raw,2,gain))
    fixtures = []
    for trial in data["trials"]:
        parts = []
        stream = bytearray(16000*2//2)
        for i,name in enumerate(trial["pieces"]):
            entry, raw = clips[name]
            start = len(stream)//2
            stream.extend(raw)
            parts.append(dict(**entry, start=start, end=len(stream)//2))
            if i<len(trial["gaps_ms"]):
                stream.extend(bytes(trial["gaps_ms"][i]*32))
        stream.extend(bytes(16000*2//3//2*2))
        with wave.open(str(output/(trial["name"]+".wav")), "wb") as f:
            f.setparams((1,2,16000,0,"NONE",""))
            f.writeframes(stream)
        fixtures.append(dict(name=trial["name"],text=" ".join(p["text"] for p in parts),language="mixed",
                             parts=parts,duration_s=len(stream)/32000,gaps_ms=trial["gaps_ms"]))
    (output/"fixtures.json").write_text(json.dumps(fixtures,ensure_ascii=False,indent=2)+"\n",encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec",type=Path)
    parser.add_argument("sources",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    generate(args.spec,args.sources,args.output)
