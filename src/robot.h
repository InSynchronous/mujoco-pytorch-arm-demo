#pragma once

#include <array>

#include <mujoco/mujoco.h>

namespace pickdemo {

inline constexpr int kArmJointCount = 4;

// Resolved indices/ranges for one actuated joint.
struct JointRef {
    int qpos = -1;
    int dof = -1;
    int actuator = -1;
    double lower = 0.0;
    double upper = 0.0;
    double ctrlLower = 0.0;
    double ctrlUpper = 0.0;
};

// Resolved indices for one prismatic gripper finger.
struct FingerRef {
    int qpos = -1;
    int dof = -1;
    int actuator = -1;
    double ctrlLower = 0.0;
    double ctrlUpper = 0.0;
};

// Handles into an mjModel/mjData for the arm used by this demo. Keeping the
// lookup out of the control code makes both sides independent of string names.
struct Arm {
    int toolSite = -1;
    std::array<JointRef, kArmJointCount> joints{};
    std::array<FingerRef, 2> fingers{};

    std::array<double, 3> toolPosition(const mjData& data) const;
};

// Resolves "yaw", "shoulder", "elbow", "wrist", "fl", "fr", "tool".
// Throws std::runtime_error naming the missing object.
Arm bindArm(const mjModel& model);

struct ServoGains {
    double kp = 600.0;
    double kv = 35.0;
};

struct GripperTravel {
    double open = 0.004;
    double closed = -0.022;
};

// Torque control: gravity/Coriolis feed-forward plus PD, clamped to each
// actuator's ctrlrange. The arm tracks joint-space targets; the fingers track
// an opening between `travel.open` and `travel.closed`.
class ArmController {
  public:
    ArmController(const Arm& arm, ServoGains armGains, ServoGains gripperGains,
                  GripperTravel travel);

    void apply(mjData& data, const std::array<double, kArmJointCount>& qReference,
               double grip) const;

  private:
    struct Channel {
        int qpos = -1;
        int dof = -1;
        int actuator = -1;
        double ctrlLower = 0.0;
        double ctrlUpper = 0.0;
    };

    std::array<Channel, kArmJointCount> armChannels_{};
    std::array<Channel, 2> fingerChannels_{};
    ServoGains armGains_{};
    ServoGains gripperGains_{};
    GripperTravel travel_{};
};

} // namespace pickdemo
