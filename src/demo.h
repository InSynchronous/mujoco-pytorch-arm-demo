#pragma once

#include <array>
#include <string>

#include "ik.h"
#include "model.h"
#include "robot.h"
#include "trajectory.h"

namespace pickdemo {

struct DemoConfig {
    std::array<double, 3> readyPosition{0.380, 0.100, 0.240};
    std::array<double, 2> cubePosition{0.380, 0.100};
    ServoGains armGains{};
    ServoGains gripperGains{400.0, 4.0};
    GripperTravel gripperTravel{};
};

// Owns the simulation of the pick-and-place task. No rendering, no I/O: the
// caller decides how to present or record it.
class PickPlaceDemo {
  public:
    explicit PickPlaceDemo(std::string modelPath = "scene.xml",
                           DemoConfig config = {});
    PickPlaceDemo(const PickPlaceDemo&) = delete;
    PickPlaceDemo& operator=(const PickPlaceDemo&) = delete;

    const mjModel& model() const { return *model_; }
    const mjData& data() const { return *data_; }
    mjData& data() { return *data_; }
    const Arm& arm() const { return arm_; }
    const Trajectory& trajectory() const { return trajectory_; }

    double time() const { return data_->time; }
    double timestep() const { return model_->opt.timestep; }
    double duration() const { return trajectory_.duration(); }

    // Advances one control tick: sample the trajectory at the current time,
    // solve IK, apply PD torques, integrate. Returns the setpoint used.
    const Setpoint& step();

    // Runs control ticks until the simulation time reaches `time`.
    void advanceTo(double time);

    std::array<double, 3> toolPosition() const;
    std::array<double, 3> cubePosition() const;
    std::array<double, 3> initialToolError() const { return initialToolError_; }

  private:
    void placeCube();

    DemoConfig config_;
    ModelPtr model_;
    DataPtr data_;
    Arm arm_;
    IkSolver ik_;
    ArmController controller_;
    Trajectory trajectory_;
    std::array<double, kArmJointCount> qCommand_{};
    std::array<double, 3> initialToolError_{};
    int cubeQpos_ = -1;
    Setpoint setpoint_;
};

} // namespace pickdemo
