#include "demo.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace pickdemo {
namespace {

const std::array<double, kArmJointCount> kInitialGuess = {0.0, -1.0, 1.4,
                                                          1.5};

} // namespace

PickPlaceDemo::PickPlaceDemo(std::string modelPath, DemoConfig config)
    : config_(std::move(config)),
      model_(loadModel(modelPath)),
      data_(makeData(*model_)),
      arm_(bindArm(*model_)),
      ik_(*model_, arm_),
      controller_(arm_, config_.armGains, config_.gripperGains,
                  config_.gripperTravel),
      trajectory_(makePickPlaceTrajectory(config_.readyPosition)) {
    const int cubeJoint = mj_name2id(model_.get(), mjOBJ_JOINT, "cube_joint");
    if (cubeJoint < 0) {
        throw std::runtime_error("model is missing joint 'cube_joint'");
    }
    cubeQpos_ = model_->jnt_qposadr[cubeJoint];

    // Put the arm at its start pose using the same IK as the run loop.
    qCommand_ = kInitialGuess;
    ik_.solve(*model_, config_.readyPosition, qCommand_, 300);
    for (int i = 0; i < kArmJointCount; ++i) {
        data_->qpos[arm_.joints[i].qpos] = qCommand_[i];
    }
    data_->qpos[arm_.fingers[0].qpos] = config_.gripperTravel.open;
    data_->qpos[arm_.fingers[1].qpos] = config_.gripperTravel.open;

    placeCube();
    mj_forward(model_.get(), data_.get());

    const std::array<double, 3> tool = toolPosition();
    for (int axis = 0; axis < 3; ++axis) {
        initialToolError_[axis] = config_.readyPosition[axis] - tool[axis];
    }
    setpoint_ = trajectory_.sample(0.0);
}

void PickPlaceDemo::placeCube() {
    const int cubeGeom = mj_name2id(model_.get(), mjOBJ_GEOM, "cube");
    const double halfHeight =
        cubeGeom >= 0 ? model_->geom_size[3 * cubeGeom + 2] : 0.02;
    const double x = config_.cubePosition[0];
    const double y = config_.cubePosition[1];
    // The pads close along the hand's local y, which rotates with the yaw
    // joint. Yaw the cube to match so the pads meet its faces.
    const double yaw = std::atan2(y, x);

    double* cube = data_->qpos + cubeQpos_;
    cube[0] = x;
    cube[1] = y;
    cube[2] = halfHeight + 5e-4; // settle onto the table
    cube[3] = std::cos(0.5 * yaw);
    cube[4] = 0.0;
    cube[5] = 0.0;
    cube[6] = std::sin(0.5 * yaw);
}

const Setpoint& PickPlaceDemo::step() {
    setpoint_ = trajectory_.sample(time());
    ik_.solve(*model_, setpoint_.position, qCommand_);
    controller_.apply(*data_, qCommand_, setpoint_.grip);
    mj_step(model_.get(), data_.get());
    return setpoint_;
}

void PickPlaceDemo::advanceTo(double time) {
    while (data_->time + 1e-9 < time) {
        step();
    }
}

std::array<double, 3> PickPlaceDemo::toolPosition() const {
    return arm_.toolPosition(*data_);
}

std::array<double, 3> PickPlaceDemo::cubePosition() const {
    const double* cube = data_->qpos + cubeQpos_;
    return {cube[0], cube[1], cube[2]};
}

} // namespace pickdemo
