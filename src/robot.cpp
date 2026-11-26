#include "robot.h"

#include <stdexcept>
#include <string>

#include "math_util.h"

namespace pickdemo {
namespace {

int requireId(const mjModel& model, mjtObj type, const char* name,
              const char* kind) {
    const int id = mj_name2id(&model, type, name);
    if (id < 0) {
        throw std::runtime_error(std::string("model is missing ") + kind + " '" +
                                 name + "'");
    }
    return id;
}

} // namespace

std::array<double, 3> Arm::toolPosition(const mjData& data) const {
    const double* position = data.site_xpos + 3 * toolSite;
    return {position[0], position[1], position[2]};
}

Arm bindArm(const mjModel& model) {
    const char* jointNames[kArmJointCount] = {"yaw", "shoulder", "elbow",
                                              "wrist"};
    const char* actuatorNames[kArmJointCount] = {"a_yaw", "a_shoulder",
                                                 "a_elbow", "a_wrist"};
    const char* fingerJoints[2] = {"fl", "fr"};
    const char* fingerActuators[2] = {"a_fl", "a_fr"};

    Arm arm;
    arm.toolSite = requireId(model, mjOBJ_SITE, "tool", "site");

    for (int i = 0; i < kArmJointCount; ++i) {
        const int joint = requireId(model, mjOBJ_JOINT, jointNames[i], "joint");
        const int actuator =
            requireId(model, mjOBJ_ACTUATOR, actuatorNames[i], "actuator");
        if (!model.jnt_limited[joint] || !model.actuator_ctrllimited[actuator]) {
            throw std::runtime_error(
                std::string("joint '") + jointNames[i] +
                "' needs a joint range and an actuator ctrlrange");
        }
        JointRef& ref = arm.joints[i];
        ref.qpos = model.jnt_qposadr[joint];
        ref.dof = model.jnt_dofadr[joint];
        ref.actuator = actuator;
        ref.lower = model.jnt_range[2 * joint + 0];
        ref.upper = model.jnt_range[2 * joint + 1];
        ref.ctrlLower = model.actuator_ctrlrange[2 * actuator + 0];
        ref.ctrlUpper = model.actuator_ctrlrange[2 * actuator + 1];
    }

    for (int i = 0; i < 2; ++i) {
        const int joint =
            requireId(model, mjOBJ_JOINT, fingerJoints[i], "joint");
        const int actuator =
            requireId(model, mjOBJ_ACTUATOR, fingerActuators[i], "actuator");
        if (!model.actuator_ctrllimited[actuator]) {
            throw std::runtime_error(std::string("actuator '") +
                                     fingerActuators[i] +
                                     "' needs a ctrlrange");
        }
        FingerRef& ref = arm.fingers[i];
        ref.qpos = model.jnt_qposadr[joint];
        ref.dof = model.jnt_dofadr[joint];
        ref.actuator = actuator;
        ref.ctrlLower = model.actuator_ctrlrange[2 * actuator + 0];
        ref.ctrlUpper = model.actuator_ctrlrange[2 * actuator + 1];
    }

    return arm;
}

ArmController::ArmController(const Arm& arm, ServoGains armGains,
                             ServoGains gripperGains, GripperTravel travel)
    : armGains_(armGains), gripperGains_(gripperGains), travel_(travel) {
    for (int i = 0; i < kArmJointCount; ++i) {
        const JointRef& joint = arm.joints[i];
        armChannels_[i] = {joint.qpos, joint.dof, joint.actuator,
                           joint.ctrlLower, joint.ctrlUpper};
    }
    for (int i = 0; i < 2; ++i) {
        const FingerRef& finger = arm.fingers[i];
        fingerChannels_[i] = {finger.qpos, finger.dof, finger.actuator,
                              finger.ctrlLower, finger.ctrlUpper};
    }
}

namespace {

void applyServo(mjData& data, int qpos, int dof, int actuator, double target,
                ServoGains gains, double ctrlLower, double ctrlUpper) {
    const double torque = data.qfrc_bias[dof] +
                          gains.kp * (target - data.qpos[qpos]) -
                          gains.kv * data.qvel[dof];
    data.ctrl[actuator] = clamp(torque, ctrlLower, ctrlUpper);
}

} // namespace

void ArmController::apply(
    mjData& data, const std::array<double, kArmJointCount>& qReference,
    double grip) const {
    grip = clamp(grip, 0.0, 1.0);

    for (int i = 0; i < kArmJointCount; ++i) {
        const Channel& channel = armChannels_[i];
        applyServo(data, channel.qpos, channel.dof, channel.actuator,
                   qReference[i], armGains_, channel.ctrlLower,
                   channel.ctrlUpper);
    }

    const double fingerTarget =
        travel_.open + (travel_.closed - travel_.open) * grip;
    for (const Channel& channel : fingerChannels_) {
        applyServo(data, channel.qpos, channel.dof, channel.actuator,
                   fingerTarget, gripperGains_, channel.ctrlLower,
                   channel.ctrlUpper);
    }
}

} // namespace pickdemo
