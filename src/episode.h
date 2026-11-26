#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>

#include "ik.h"
#include "model.h"
#include "robot.h"
#include "trajectory.h"

namespace pickdemo {

// Arm joints and two gripper fingers, in a fixed order.
inline constexpr int kMotorCount = kArmJointCount + 2;

struct EpisodeConfig {
    PickPlacePath path{};
    ServoGains armGains{};
    ServoGains gripperGains{400.0, 4.0};
    GripperTravel gripperTravel{};
    double controlDt = 0.1; // seconds between observations/actions

    // Cube sampling: a rectangle on the table, an annulus around the base,
    // and a clear zone around the place target.
    double cubeXMin = 0.220;
    double cubeXMax = 0.470;
    double cubeYMin = -0.160;
    double cubeYMax = 0.240;
    double minRadius = 0.280;
    double maxRadius = 0.470;
    double placeClearance = 0.150;

    std::uint32_t seed = 1;
};

// One recorded control tick. `q`/`qdot` are measured before the step; `action`
// is the joint change that the expert commands for the next interval.
struct Transition {
    std::array<double, kMotorCount> q{};
    std::array<double, kMotorCount> qdot{};
    std::array<double, kMotorCount> action{};
    std::array<double, 2> cube{}; // logged for analysis only
    std::string label;
    int phase = 0;
};

// A single randomized pick-and-place demonstration. The arm always starts at
// the same home pose; only the cube position changes, so a vision policy
// cannot recover the cube location from the initial joint state.
class PickPlaceEpisode {
  public:
    PickPlaceEpisode(std::string modelPath, EpisodeConfig config);

    void reset(std::uint32_t episodeIndex);

    // Observes the current state, computes the next expert command, applies it
    // for one control interval, and returns the pre-step transition.
    const Transition& step();

    int ticks() const { return plannedTicks_; }
    double duration() const { return plannedTicks_ * controlDt_; }
    double controlDt() const { return controlDt_; }
    double simDt() const { return model_->opt.timestep; }

    const mjModel& model() const { return *model_; }
    const mjData& data() const { return *data_; }
    mjData& data() { return *data_; }
    const std::array<double, kMotorCount>& homeQpos() const { return homeQ_; }

    std::array<double, 2> cubePosition() const;
    bool succeeded() const;

  private:
    void sampleCube(std::uint32_t episodeIndex);
    void resetArm();
    std::array<double, kMotorCount> measureQ() const;
    std::array<double, kMotorCount> measureQdot() const;

    EpisodeConfig config_;
    ModelPtr model_;
    DataPtr data_;
    Arm arm_;
    IkSolver ik_;
    ArmController controller_;
    std::array<double, kArmJointCount> homeArmQ_{};
    std::array<double, kMotorCount> homeQ_{};
    std::array<double, kArmJointCount> qcmdArm_{};
    std::optional<Trajectory> trajectory_;
    Transition lastTransition_;

    int cubeQpos_ = -1;
    int stepsPerTick_ = 1;
    int plannedTicks_ = 0;
    double controlDt_ = 0.1;
    int tick_ = 0;
    double maxCubeZ_ = 0.0;
    double cubeX_ = 0.0;
    double cubeY_ = 0.0;
};

} // namespace pickdemo
