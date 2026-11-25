#pragma once
#include "layer.hpp"
#ifdef FZ_AREA_REVISION
#include "ratio.hpp"
#else
namespace fzint {
inline uint32_t ratio(uint32_t d) { return d; }
inline int32_t scale_code(int32_t x,uint32_t d) { return int32_t(divide(x,d)); }
inline int64_t wide_ratio(int64_t x,uint32_t d) { return divide(x,d); }
}
#endif
