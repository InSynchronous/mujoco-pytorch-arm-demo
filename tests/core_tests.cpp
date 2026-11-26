#include <cmath>
#include <cstdio>
#include <string>

#include "ik.h"
#include "model.h"
#include "robot.h"
#include "trajectory.h"

using namespace pickdemo;

namespace {

int gFailures = 0;

#define CHECK(condition)                                                      \
    do {                                                                      \
        if (!(condition)) {                                                   \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,      \
                         #condition);                                         \
            ++gFailures;                                                      \
        }                                                                     \
    } while (0)

#define CHECK_NEAR(actual, expected, tolerance)                               \
    do {                                                                      \
        const double actualValue = (actual);                                  \
        const double expectedValue = (expected);                              \
        const double toleranceValue = (tolerance);                            \
        if (std::fabs(actualValue - expectedValue) > toleranceValue) {        \
            std::fprintf(stderr,                                              \
                         "FAIL %s:%d: %.6g != %.6g (tol %.1g)\n", __FILE__,   \
                         __LINE__, actualValue, expectedValue,                \
                         toleranceValue);                                     \
            ++gFailures;                                                      \
        }                                                                     \
    } while (0)

void testTrajectory() {
    const std::array<double, 3> start{0.38, 0.10, 0.24};
    const Trajectory trajectory = makePickPlaceTrajectory(start);

    CHECK_NEAR(trajectory.duration(), 10.15, 1e-9);

    // Before the first phase the arm holds its start pose.
    const Setpoint begin = trajectory.sample(-1.0);
    CHECK(std::string(begin.label) == "READY");
    CHECK_NEAR(begin.position[0], start[0], 1e-12);
    CHECK_NEAR(begin.position[2], start[2], 1e-12);
    CHECK_NEAR(begin.grip, 0.0, 1e-12);

    // Smoothstep midpoint of APPROACH (0.60 .. 1.60).
    const Setpoint midApproach = trajectory.sample(1.10);
    CHECK(std::string(midApproach.label) == "APPROACH");
    CHECK_NEAR(midApproach.position[2], 0.5 * (start[2] + 0.024), 1e-9);

    // Half way through the first closing phase, the grip is half closed.
    const Setpoint halfClosed = trajectory.sample(1.60 + 0.5 * 0.45);
    CHECK(std::string(halfClosed.label) == "GRASP");
    CHECK_NEAR(halfClosed.grip, 0.5, 1e-9);
    CHECK_NEAR(halfClosed.position[2], 0.024, 1e-12);

    // LIFT must move upward from the grasp height.
    const Setpoint lifting = trajectory.sample(3.0);
    CHECK(std::string(lifting.label) == "LIFT");
    CHECK(lifting.position[2] > 0.024);
    CHECK_NEAR(lifting.grip, 1.0, 1e-12);

    // Past the end, everything clamps to the final pose.
    const Setpoint end = trajectory.sample(trajectory.duration() + 5.0);
    CHECK(std::string(end.label) == "DONE");
    CHECK_NEAR(end.position[0], 0.330, 1e-12);
    CHECK_NEAR(end.position[1], 0.020, 1e-12);
    CHECK_NEAR(end.position[2], 0.380, 1e-12);
    CHECK_NEAR(end.grip, 0.0, 1e-12);
}

void testInverseKinematics(const std::string& scenePath) {
    ModelPtr model = loadModel(scenePath);
    DataPtr data = makeData(*model);
    const Arm arm = bindArm(*model);
    IkSolver ik(*model, arm);

    std::array<double, 4> q{0.0, -1.0, 1.4, 1.5};
    const std::array<double, 3> targets[] = {
        {0.380, 0.100, 0.300},
        {0.380, 0.100, 0.024},
        {0.120, -0.320, 0.200},
    };

    for (const auto& target : targets) {
        ik.solve(*model, target, q, 300);

        for (int i = 0; i < kArmJointCount; ++i) {
            data->qpos[arm.joints[i].qpos] = q[i];
        }
        mj_kinematics(model.get(), data.get());

        const auto reached = arm.toolPosition(*data);
        CHECK_NEAR(reached[0], target[0], 1e-3);
        CHECK_NEAR(reached[1], target[1], 1e-3);
        CHECK_NEAR(reached[2], target[2], 1e-3);

        // The tool x-axis (column 0 of the site frame) must point down.
        const double* rotation = data->site_xmat + 9 * arm.toolSite;
        CHECK(std::fabs(rotation[0]) < 1e-3);
        CHECK(std::fabs(rotation[3]) < 1e-3);
        CHECK_NEAR(rotation[6], -1.0, 1e-3);
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string scene =
        argc > 1 ? argv[1] : std::string("scene.xml");
    try {
        testTrajectory();
        testInverseKinematics(scene);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }

    if (gFailures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", gFailures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
