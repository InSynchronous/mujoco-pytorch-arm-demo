#pragma once

#include <algorithm>

namespace pickdemo {

template <typename T>
constexpr T clamp(T value, T low, T high) {
    return std::min(high, std::max(low, value));
}

// Smooth, C1-continuous ramp from 0 to 1 (zero slope at both ends).
inline double smoothstep(double u) {
    u = clamp(u, 0.0, 1.0);
    return u * u * (3.0 - 2.0 * u);
}

} // namespace pickdemo
