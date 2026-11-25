#pragma once

#include "mac_array.hpp"
#include "tile_pipeline.hpp"
#include <hls_stream.h>

namespace edge_speech::pl {

// Experimental T014 raw interface. Exactly m*32 row-major packets are emitted
// for a valid descriptor. Rejected pairs emit invalid zero packets; consumers
// must retain tile-level transactional commit, as in tile_pipeline.
struct MacPingPongFour {
    int32_t dot[4];
    uint32_t issue_steps;
    bool valid;
};

// One locked K<=64 weight group and m rows of four packed activation beats.
// The caller owns the weight-group lock through completion of this function.
MacStatus mac_pingpong_group(MacMode mode, const TileBeat* activation,
                             const uint64_t weight[][16], MacExtent extent,
                             hls::stream<MacPingPongFour>& partials,
                             MacAccounting& accounting);
}

uint8_t mac_pingpong(uint8_t mode, uint16_t m, uint16_t n, uint16_t k,
                     const edge_speech::pl::TileBeat activation[64],
                     const uint64_t weight[64][16],
                     hls::stream<edge_speech::pl::MacPingPongFour>& partials,
                     uint32_t completion[6]);
