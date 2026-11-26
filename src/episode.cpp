#include "episode.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

#include "math_util.h"

namespace pickdemo {
namespace {

const std::array<double, kArmJointCount> kInitialGuess = {0.0, -1.0, 1.4,
                                                          1.5};

} // namespace

PickPlaceEpisode::PickPlaceEpisode(std::string modelPath,
                                   EpisodeConfig config)
    : config_(std::move(config)), model_(loadModel(modelPath)),
      data_(makeData(*model_)), arm_(bindArm(*model_)),
      ik_(*model_, arm_),
      controller_(arm_, config_.armGains, config_.gripperGains,
                  config_.gripperTravel) {
    const int cubeJoint = mj_name2id(model_.get(), mjOBJ_JOINT, "cube_joint");
    if (cubeJoint < 0) {
        throw std::runtime_error("model is missing joint 'cube_joint'");
    }
    cubeQpos_ = model_->jnt_qposadr[cubeJoint];

    stepsPerTick_ = static_cast<int>(std::lround(
        config_.controlDt / model_->opt.timestep));
    if (stepsPerTick_ < 1) {
        throw std::invalid_argument("control interval is shorter than one step");
    }
    controlDt_ = stepsPerTick_ * model_->opt.timestep;
    plannedTicks_ = static_cast<int>(std::lround(
        makePickPlaceTrajectory(config_.path).duration() / controlDt_));

    // One fixed home pose for every episode.
    homeArmQ_ = kInitialGuess;
    ik_.solve(*model_, config_.path.home, homeArmQ_, 300);
    homeQ_ = {homeArmQ_[0], homeArmQ_[1], homeArmQ_[2], homeArmQ_[3],
              config_.gripperTravel.open, config_.gripperTravel.open};
}

void PickPlaceEpisode::sampleCube(std::uint32_t episodeIndex) {
    std::mt19937 rng(config_.seed +
                     episodeIndex * static_cast<std::uint32_t>(0x9e3779b9u));
    std::uniform_real_distribution<double> distX(config_.cubeXMin,
                                                 config_.cubeXMax);
    std::uniform_real_distribution<double> distY(config_.cubeYMin,
                                                 config_.cubeYMax);

    for (int attempt = 0; attempt < 10000; ++attempt) {
        const double x = distX(rng);
        const double y = distY(rng);
        const double radius = std::hypot(x, y);
        if (radius < config_.minRadius || radius > config_.maxRadius) {
            continue;
        }
        const double toPlace =
            std::hypot(x - config_.path.place[0], y - config_.path.place[1]);
        if (toPlace < config_.placeClearance) {
            continue;
        }
        cubeX_ = x;
        cubeY_ = y;
        return;
    }
    throw std::runtime_error("could not sample a valid cube position");
}

void PickPlaceEpisode::resetArm() {
    for (int i = 0; i < kArmJointCount; ++i) {
        data_->qpos[arm_.joints[i].qpos] = homeArmQ_[i];
    }
    for (int f = 0; f < 2; ++f) {
        data_->qpos[arm_.fingers[f].qpos] = config_.gripperTravel.open;
    }
}

void PickPlaceEpisode::reset(std::uint32_t episodeIndex) {
    mj_resetData(model_.get(), data_.get());

    sampleCube(episodeIndex);
    resetArm();

    const int cubeGeom = mj_name2id(model_.get(), mjOBJ_GEOM, "cube");
    const double halfHeight =
        cubeGeom >= 0 ? model_->geom_size[3 * cubeGeom + 2] : 0.02;
    const double yaw = std::atan2(cubeY_, cubeX_);
    double* cube = data_->qpos + cubeQpos_;
    cube[0] = cubeX_;
    cube[1] = cubeY_;
    cube[2] = halfHeight + 5e-4;
    cube[3] = std::cos(0.5 * yaw);
    cube[4] = 0.0;
    cube[5] = 0.0;
    cube[6] = std::sin(0.5 * yaw);

    PickPlacePath path = config_.path;
    path.pick[0] = cubeX_;
    path.pick[1] = cubeY_;
    path.grasp[0] = cubeX_;
    path.grasp[1] = cubeY_;
    trajectory_.emplace(makePickPlaceTrajectory(path));

    mj_forward(model_.get(), data_.get());
    qcmdArm_ = homeArmQ_;
    tick_ = 0;
    maxCubeZ_ = cube[2];
    lastTransition_ = Transition{};
}

std::array<double, kMotorCount> PickPlaceEpisode::measureQ() const {
    std::array<double, kMotorCount> q{};
    for (int i = 0; i < kArmJointCount; ++i) {
        q[i] = data_->qpos[arm_.joints[i].qpos];
    }
    for (int f = 0; f < 2; ++f) {
        q[kArmJointCount + f] = data_->qpos[arm_.fingers[f].qpos];
    }
    return q;
}

std::array<double, kMotorCount> PickPlaceEpisode::measureQdot() const {
    std::array<double, kMotorCount> qd{};
    for (int i = 0; i < kArmJointCount; ++i) {
        qd[i] = data_->qvel[arm_.joints[i].dof];
    }
    for (int f = 0; f < 2; ++f) {
        qd[kArmJointCount + f] = data_->qvel[arm_.fingers[f].dof];
    }
    return qd;
}

const Transition& PickPlaceEpisode::step() {
    if (!trajectory_) {
        throw std::logic_error("PickPlaceEpisode::step called before reset");
    }

    Transition transition;
    transition.q = measureQ();
    transition.qdot = measureQdot();
    transition.cube = {cubeX_, cubeY_};

    // The action is the joint change that would put the motors at the next
    // expert target. Using the measured q (rather than an internal target)
    // keeps the policy Markov in the observation.
    const double nextTime = (tick_ + 1) * controlDt_;
    const Setpoint setpoint = trajectory_->sample(nextTime);
    std::array<double, kArmJointCount> nextArm = qcmdArm_;
    ik_.solve(*model_, setpoint.position, nextArm);
    const double grip = clamp(setpoint.grip, 0.0, 1.0);
    const double fingerTarget =
        config_.gripperTravel.open +
        (config_.gripperTravel.closed - config_.gripperTravel.open) * grip;

    for (int i = 0; i < kArmJointCount; ++i) {
        transition.action[i] = nextArm[i] - transition.q[i];
    }
    transition.action[kArmJointCount + 0] =
        fingerTarget - transition.q[kArmJointCount + 0];
    transition.action[kArmJointCount + 1] =
        fingerTarget - transition.q[kArmJointCount + 1];
    transition.label = setpoint.label;
    transition.phase = setpoint.phase;

    for (int step = 0; step < stepsPerTick_; ++step) {
        controller_.apply(*data_, nextArm, grip);
        mj_step(model_.get(), data_.get());
    }

    qcmdArm_ = nextArm;
    maxCubeZ_ = std::max(maxCubeZ_, data_->qpos[cubeQpos_ + 2]);
    ++tick_;
    lastTransition_ = std::move(transition);
    return lastTransition_;
}

std::array<double, 2> PickPlaceEpisode::cubePosition() const {
    return {data_->qpos[cubeQpos_], data_->qpos[cubeQpos_ + 1]};
}

bool PickPlaceEpisode::succeeded() const {
    const auto cube = cubePosition();
    const double distance =
        std::hypot(cube[0] - config_.path.place[0],
                   cube[1] - config_.path.place[1]);
    return distance < 0.05 && data_->qpos[cubeQpos_ + 2] < 0.05 &&
           maxCubeZ_ > 0.08;
}

} // namespace pickdemo
