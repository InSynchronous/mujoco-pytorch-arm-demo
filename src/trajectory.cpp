#include "trajectory.h"

#include <cmath>
#include <stdexcept>
#include <utility>

#include "math_util.h"

namespace pickdemo {

Trajectory::Trajectory(std::vector<Phase> phases, std::array<double, 3> start)
    : start_(start), phases_(std::move(phases)) {
    if (phases_.empty()) {
        throw std::invalid_argument("trajectory needs at least one phase");
    }
    for (const Phase& phase : phases_) {
        if (!(phase.duration > 0.0)) {
            throw std::invalid_argument("phase '" + phase.label +
                                        "' has a non-positive duration");
        }
        duration_ += phase.duration;
    }
}

Setpoint Trajectory::sample(double time) const {
    Setpoint out;
    out.position = start_;
    out.grip = 0.0;

    double phaseStart = 0.0;
    for (size_t i = 0; i < phases_.size(); ++i) {
        const Phase& phase = phases_[i];
        const bool isLast = (i + 1 == phases_.size());

        if (time <= phaseStart + phase.duration || isLast) {
            const double blend =
                smoothstep((time - phaseStart) / phase.duration);
            for (int axis = 0; axis < 3; ++axis) {
                out.position[axis] +=
                    (phase.position[axis] - out.position[axis]) * blend;
            }
            out.grip += (phase.grip - out.grip) * blend;
            out.label = phase.label;
            out.phase = static_cast<int>(i);
            return out;
        }

        out.position = phase.position;
        out.grip = phase.grip;
        phaseStart += phase.duration;
    }

    // The loop always returns above (the last phase is forced).
    out.label = phases_.back().label;
    out.phase = static_cast<int>(phases_.size()) - 1;
    return out;
}

Trajectory makePickPlaceTrajectory(const PickPlacePath& path) {
    const std::array<double, 3> lift = {path.pick[0], path.pick[1], 0.280};
    const std::array<double, 3> placeLift = {path.place[0], path.place[1],
                                             0.280};
    return Trajectory(
        {
            {0.20, path.home, 0.0, "HOLD"},
            {1.40, path.pick, 0.0, "APPROACH"},
            {0.80, path.grasp, 0.0, "DESCEND"},
            {0.50, path.grasp, 1.0, "GRASP"},
            {0.90, lift, 1.0, "LIFT"},
            {1.50, placeLift, 1.0, "TRANSPORT"},
            {0.90, path.place, 1.0, "PLACE"},
            {0.50, path.place, 0.0, "RELEASE"},
            {0.90, path.retreat, 0.0, "RETREAT"},
            {0.80, path.home, 0.0, "HOME"},
            {0.20, path.home, 0.0, "DONE"},
        },
        path.home);
}

Trajectory makePickPlaceTrajectory(const std::array<double, 3>& start) {
    return Trajectory(
        {
            {0.60, start, 0.0, "READY"},
            {1.00, {0.380, 0.100, 0.024}, 0.0, "APPROACH"},
            {0.45, {0.380, 0.100, 0.024}, 1.0, "GRASP"},
            {0.40, {0.380, 0.100, 0.024}, 1.0, "GRASP"},
            {1.30, {0.380, 0.100, 0.310}, 1.0, "LIFT"},
            {1.80, {0.120, -0.320, 0.320}, 1.0, "TRANSPORT"},
            {1.00, {0.120, -0.320, 0.024}, 1.0, "PLACE"},
            {0.45, {0.120, -0.320, 0.024}, 0.0, "RELEASE"},
            {0.35, {0.120, -0.320, 0.024}, 0.0, "RELEASE"},
            {1.00, {0.190, -0.100, 0.360}, 0.0, "RETREAT"},
            {1.00, {0.330, 0.020, 0.380}, 0.0, "HOME"},
            {0.80, {0.330, 0.020, 0.380}, 0.0, "DONE"},
        },
        start);
}

} // namespace pickdemo
