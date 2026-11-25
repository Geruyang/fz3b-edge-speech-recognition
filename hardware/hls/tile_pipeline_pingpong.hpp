#pragma once

#include "tile_pipeline.hpp"

// Isolated offline comparison top; descriptor and numerical contract match tile_pipeline.
void tile_pp(const edge_speech::pl::TileBeat* hp0, const edge_speech::pl::TileBeat* hp1,
                   const edge_speech::pl::TileBeat* hp2, const edge_speech::pl::TileBeat* hp3,
                   const edge_speech::pl::TileBeat* activation, const float* row_scales,
                   const float* bias, float* output, uint16_t m, uint16_t n, uint16_t k,
                   uint32_t weight_base, uint32_t scale_base, uint32_t tile_id, bool zero_bias,
                   uint8_t active_hp_mask, uint64_t profile_epoch, uint64_t completion[9]);
