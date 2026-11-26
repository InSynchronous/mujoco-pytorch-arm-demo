#include "ik.h"

#include <cmath>
#include <stdexcept>

#include "math_util.h"

namespace pickdemo {
namespace {

using Matrix6 = std::array<std::array<double, 6>, 6>;
using Vector6 = std::array<double, 6>;

std::array<double, 3> cross(const std::array<double, 3>& a,
                            const std::array<double, 3>& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}

// Gaussian elimination with partial pivoting. A and b are consumed.
bool solveLinear(Matrix6 a, Vector6 b, Vector6& x) {
    for (int col = 0; col < 6; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 6; ++row) {
            if (std::fabs(a[row][col]) > std::fabs(a[pivot][col])) {
                pivot = row;
            }
        }
        if (std::fabs(a[pivot][col]) < 1e-12) {
            return false;
        }
        if (pivot != col) {
            std::swap(a[col], a[pivot]);
            std::swap(b[col], b[pivot]);
        }
        for (int row = col + 1; row < 6; ++row) {
            const double factor = a[row][col] / a[col][col];
            for (int k = col; k < 6; ++k) {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    for (int row = 5; row >= 0; --row) {
        double sum = b[row];
        for (int k = row + 1; k < 6; ++k) {
            sum -= a[row][k] * x[k];
        }
        x[row] = sum / a[row][row];
    }
    return true;
}

} // namespace

IkSolver::IkSolver(const mjModel& model, const Arm& arm)
    : IkSolver(model, arm, Settings{}) {}

IkSolver::IkSolver(const mjModel& model, const Arm& arm, Settings settings)
    : settings_(settings), scratch_(makeData(model)) {
    if (arm.toolSite < 0) {
        throw std::invalid_argument("IK: arm has no tool site");
    }
    if (settings_.iterations < 1 || !(settings_.damping > 0.0) ||
        settings_.maxJointStep <= 0.0 || settings_.positionWeight < 0.0 ||
        settings_.rotationWeight < 0.0) {
        throw std::invalid_argument("IK: invalid settings");
    }

    site_ = arm.toolSite;
    nv_ = model.nv;
    for (int i = 0; i < kArmJointCount; ++i) {
        channels_[i] = {arm.joints[i].qpos, arm.joints[i].dof,
                        arm.joints[i].lower, arm.joints[i].upper};
    }
    jacPosition_.assign(3 * nv_, 0.0);
    jacRotation_.assign(3 * nv_, 0.0);
}

void IkSolver::solve(const mjModel& model, const std::array<double, 3>& target,
                     std::array<double, kArmJointCount>& q) {
    solve(model, target, q, settings_.iterations);
}

void IkSolver::solve(const mjModel& model, const std::array<double, 3>& target,
                     std::array<double, kArmJointCount>& q, int iterations) {
    std::array<double, 3> approach = settings_.approachDirection;
    const double approachNorm = std::sqrt(
        approach[0] * approach[0] + approach[1] * approach[1] +
        approach[2] * approach[2]);
    if (approachNorm > 1e-9) {
        for (double& component : approach) {
            component /= approachNorm;
        }
    } else {
        approach = {0.0, 0.0, -1.0};
    }

    const double positionWeight = settings_.positionWeight;
    const double rotationWeight = settings_.rotationWeight;
    const double dampingSquared = settings_.damping * settings_.damping;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (int i = 0; i < kArmJointCount; ++i) {
            scratch_->qpos[channels_[i].qpos] = q[i];
        }
        mj_kinematics(&model, scratch_.get());
        mj_comPos(&model, scratch_.get()); // composite DOFs, needed by jacSite
        mj_jacSite(&model, scratch_.get(), jacPosition_.data(),
                   jacRotation_.data(), site_);

        const double* sitePosition = scratch_->site_xpos + 3 * site_;
        const double* siteRotation = scratch_->site_xmat + 9 * site_;
        // Column 0 of the site rotation is the tool's approach axis in world.
        const std::array<double, 3> approachNow = {
            siteRotation[0], siteRotation[3], siteRotation[6]};
        const std::array<double, 3> orientationError =
            cross(approachNow, approach);

        const Vector6 error = {
            positionWeight * (target[0] - sitePosition[0]),
            positionWeight * (target[1] - sitePosition[1]),
            positionWeight * (target[2] - sitePosition[2]),
            rotationWeight * orientationError[0],
            rotationWeight * orientationError[1],
            rotationWeight * orientationError[2],
        };

        std::array<std::array<double, kArmJointCount>, 6> jacobian{};
        for (int axis = 0; axis < kArmJointCount; ++axis) {
            const int dof = channels_[axis].dof;
            jacobian[0][axis] = positionWeight * jacPosition_[0 * nv_ + dof];
            jacobian[1][axis] = positionWeight * jacPosition_[1 * nv_ + dof];
            jacobian[2][axis] = positionWeight * jacPosition_[2 * nv_ + dof];
            jacobian[3][axis] = rotationWeight * jacRotation_[0 * nv_ + dof];
            jacobian[4][axis] = rotationWeight * jacRotation_[1 * nv_ + dof];
            jacobian[5][axis] = rotationWeight * jacRotation_[2 * nv_ + dof];
        }

        Matrix6 normalEquations{};
        for (int row = 0; row < 6; ++row) {
            for (int col = 0; col < 6; ++col) {
                double sum = 0.0;
                for (int axis = 0; axis < kArmJointCount; ++axis) {
                    sum += jacobian[row][axis] * jacobian[col][axis];
                }
                normalEquations[row][col] =
                    sum + (row == col ? dampingSquared : 0.0);
            }
        }

        Vector6 y{};
        if (!solveLinear(normalEquations, error, y)) {
            return;
        }

        for (int axis = 0; axis < kArmJointCount; ++axis) {
            double step = 0.0;
            for (int row = 0; row < 6; ++row) {
                step += jacobian[row][axis] * y[row];
            }
            step = clamp(step, -settings_.maxJointStep, settings_.maxJointStep);
            q[axis] = clamp(q[axis] + step, channels_[axis].lower,
                            channels_[axis].upper);
        }
    }
}

} // namespace pickdemo
