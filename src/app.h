#pragma once

#include "options.h"

namespace pickdemo {

// Runs the still or video mode described by `options`.
// Returns 0 on success; throws on setup or encoder failures.
int run(const Options& options);

} // namespace pickdemo
