#pragma once

#include <array>
#include <memory>
#include <vector>

#include <mujoco/mujoco.h>

#include "model.h"
#include "robot.h"

namespace pickdemo {

// Damped-least-squares IK for the tool site. The task is the 3D position plus
// the orientation of the tool's local x-axis, which is the gripper approach
// direction (kept pointing down by default). The arm's four joints are the only
// variables; the fingers and any free bodies are ignored.
class IkSolver {
  public:
    struct Settings {
        std::array<double, 3> approachDirection{0.0, 0.0, -1.0};
        double positionWeight = 1.0;
        double rotationWeight = 0.8;
        double damping = 0.02;    // lambda in J J^T + lambda^2 I
        double maxJointStep = 0.15; // rad per iteration
        int iterations = 6;
    };

    IkSolver(const mjModel& model, const Arm& arm);
    IkSolver(const mjModel& model, const Arm& arm, Settings settings);
    IkSolver(const IkSolver&) = delete;
    IkSolver& operator=(const IkSolver&) = delete;

    // Updates `q` in place. Overload 3 uses settings().iterations.
    void solve(const mjModel& model, const std::array<double, 3>& target,
               std::array<double, kArmJointCount>& q);
    void solve(const mjModel& model, const std::array<double, 3>& target,
               std::array<double, kArmJointCount>& q, int iterations);

    Settings& settings() { return settings_; }
    const Settings& settings() const { return settings_; }

  private:
    struct Channel {
        int qpos = -1;
        int dof = -1;
        double lower = 0.0;
        double upper = 0.0;
    };

    std::array<Channel, kArmJointCount> channels_{};
    int site_ = -1;
    int nv_ = 0;
    Settings settings_;
    DataPtr scratch_;
    std::vector<double> jacPosition_;
    std::vector<double> jacRotation_;
};

} // namespace pickdemo
