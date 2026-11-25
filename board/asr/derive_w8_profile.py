"""Insert CPU timings into a separate copy of the shared-quant W8 reference."""

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "hardware/hls/asr_blocks_shared_quant.cpp"


def derive(raw):
    text = raw.decode().replace("\r\n", "\n")

    def replace(before, after):
        nonlocal text
        if text.count(before) != 1:
            raise ValueError("W8 reference boundary changed: " + before.splitlines()[0])
        text = text.replace(before, after)

    replace('#include "vector_ops.hpp"', '#include "vector_ops.hpp"\n#include "board/asr/w8_profile.hpp"')
    replace("namespace edge_speech::pl {\nnamespace {", """namespace edge_speech::pl {
namespace {

using asr::W8Part;
using asr::w8_clock;
using asr::w8_record;""")
    replace("                    const float result = stage.activate ? double_swish(value) : value;",
            """                    const bool measure_swish = stage.activate && !onchip && asr::current_w8_profile;
                    const auto swish_start = measure_swish ? w8_clock() : asr::W8Clock::time_point{};
                    const float result = stage.activate ? double_swish(value) : value;
                    if (measure_swish) w8_record(W8Part::ffn_double_swish, swish_start);""")
    replace("    using namespace edge_speech::pl;\n#pragma HLS ALLOCATION", """    using namespace edge_speech::pl;
    using namespace edge_speech::asr;
#pragma HLS ALLOCATION""")
    replace("    const unsigned passes = embedding ? 7 : (conv ? 3 : 2);", """    const unsigned passes = embedding ? 7 : (conv ? 3 : 2);
    const W8Part embedding_parts[] = {W8Part::embedding_conv1_scan, W8Part::embedding_conv1_encode,
        W8Part::embedding_conv2_scan, W8Part::embedding_conv2_encode,
        W8Part::embedding_conv3_scan, W8Part::embedding_conv3_encode, W8Part::embedding_projection};
    const W8Part conv_parts[] = {W8Part::conv_pw1, W8Part::conv_dw31, W8Part::conv_pw2};
    const W8Part conv_quant[] = {W8Part::conv_input_quant, W8Part::conv_depth_quant, W8Part::conv_tail_quant};""")
    replace("        float amax_to_encode = amax;", """        const auto quant_start = encode_input ? w8_clock() : W8Clock::time_point{};
        float amax_to_encode = amax;""")
    replace("            activation_scale = next_scale;\n        }\n        const bool report_raw", """            activation_scale = next_scale;
            w8_record(embedding ? W8Part::embedding_input_quant : conv ? conv_quant[i] :
                      i == 0 ? W8Part::ffn_input_quant : W8Part::ffn_hidden_quant, quant_start);
        }
        const bool report_raw""")
    replace("        const auto status = dw ? depthwise", """        const auto stage_start = w8_clock();
        const auto status = dw ? depthwise""")
    replace("        if (pass == DensePass::scan) continue;", """        w8_record(embedding ? embedding_parts[iteration] : conv ? conv_parts[i] :
                  i == 0 ? W8Part::ffn_linear1 : W8Part::ffn_linear2, stage_start);
        if (pass == DensePass::scan) continue;""")
    replace("        if (conv && i == 0) {\n            for", """        if (conv && i == 0) {
            const auto history_start = w8_clock();
            for""")
    replace("            staged_glu(final_tensor, rows);", """            w8_record(W8Part::conv_old_history, history_start);
            const auto glu_start = w8_clock();
            staged_glu(final_tensor, rows);
            w8_record(W8Part::conv_glu, glu_start);""")
    replace("    if (embedding) {\n        for (unsigned at = 0; at < 16 * asr_channels; ++at)",
            """    const auto write_start = w8_clock();
    if (embedding) {
        for (unsigned at = 0; at < 16 * asr_channels; ++at)""")
    replace("        return uint8_t(AsrStatus::ok);\n    }\n    for (unsigned at = 0; at < rows * asr_channels; ++at)",
            """        w8_record(W8Part::write_output, write_start);
        return uint8_t(AsrStatus::ok);
    }
    for (unsigned at = 0; at < rows * asr_channels; ++at)""")
    replace("    if (conv) {\n        for (unsigned at = 0; at < asr_cache_elements; ++at)",
            """    w8_record(W8Part::write_output, write_start);
    if (conv) {
        const auto history_start = w8_clock();
        for (unsigned at = 0; at < asr_cache_elements; ++at)""")
    replace("            next_cache[at] = scratch[rows * asr_channels + at];\n        }", """            next_cache[at] = scratch[rows * asr_channels + at];
        }
        w8_record(W8Part::conv_next_history, history_start);""")
    return "// CPU diagnostic, derived from SHA-256 " + hashlib.sha256(raw).hexdigest() + "\n" + text


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(derive(REFERENCE.read_bytes()), encoding="utf-8", newline="\n")
