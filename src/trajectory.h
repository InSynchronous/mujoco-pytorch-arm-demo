#pragma once

#include <array>
#include <string>
#include <string_view>
#include <vector>

namespace pickdemo {

// A tool-space setpoint: where the gripper should be and how open it is.
struct Setpoint {
    std::array<double, 3> position{}; // tool-site target in world space
    double grip = 0.0;                // 0 = open, 1 = closed
    std::string_view label;           // human-readable phase name
    int phase = 0;                    // index of the active phase
};

// One leg of the task: move to `position`, ramp the grip to `grip`, over
// `duration` seconds. The motion always starts where the previous leg ended.
struct Phase {
    double duration = 1.0;
    std::array<double, 3> position{};
    double grip = 0.0;
    std::string label;
};

// Piecewise task-space trajectory with smoothstep timing. Pure data: no
// MuJoCo dependency, so it is easy to test or replace.
class Trajectory {
  public:
    Trajectory(std::vector<Phase> phases, std::array<double, 3> start);

    double duration() const { return duration_; }
    const std::vector<Phase>& phases() const { return phases_; }

    // Sample at `time`. Times outside [0, duration] clamp to the first/last
    // pose. The returned views point into the trajectory's own phase labels.
    Setpoint sample(double time) const;

  private:
    std::array<double, 3> start_{};
    std::vector<Phase> phases_;
    double duration_ = 0.0;
};

// The sequence used by the hand-written pick-and-place demo, starting at
// `start` (the arm begins above the cube).
Trajectory makePickPlaceTrajectory(const std::array<double, 3>& start);

// The same task generalized to a home pose and a cube position. Used to build
// one expert demonstration per randomized cube placement.
struct PickPlacePath {
    std::array<double, 3> home{0.330, 0.020, 0.380};
    std::array<double, 3> pick{0.380, 0.100, 0.240}; // above the cube
    std::array<double, 3> grasp{0.380, 0.100, 0.024};
    std::array<double, 3> place{0.120, -0.320, 0.024};
    std::array<double, 3> retreat{0.190, -0.100, 0.340};
};

Trajectory makePickPlaceTrajectory(const PickPlacePath& path);

} // namespace pickdemo
