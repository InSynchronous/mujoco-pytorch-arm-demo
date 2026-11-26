#include "app.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "demo.h"
#include "renderer.h"
#include "video_writer.h"

namespace pickdemo {
namespace {

void printVec(const char* name, const std::array<double, 3>& value) {
    std::printf("%s %.4f %.4f %.4f\n", name, value[0], value[1], value[2]);
}

} // namespace

int run(const Options& options) {
    PickPlaceDemo demo(options.modelPath);

    const auto error = demo.initialToolError();
    std::printf("initial tool error: %.5f %.5f %.5f\n", error[0], error[1],
                error[2]);

    Renderer renderer(demo.model(), options.width, options.height,
                      options.title, options.camera);
    std::printf("framebuffer: %dx%d\n", renderer.width(), renderer.height());

    if (!options.stillPath.empty()) {
        const double when =
            options.stillTime >= 0.0 ? options.stillTime : 0.5 * demo.duration();
        std::printf("rendering still at t=%.2f (total %.2f)\n", when,
                    demo.duration());
        demo.advanceTo(when);
        const auto& rgb = renderer.render(demo.data(), "STILL");
        Renderer::writePpm(options.stillPath, rgb, renderer.width(),
                           renderer.height());
        printVec("still tool", demo.toolPosition());
        printVec("still cube", demo.cubePosition());
        return 0;
    }

    // Physics runs at the model timestep; a video frame is an integer number
    // of those steps. The resulting fps may differ slightly from the request
    // if the timestep does not divide 1/fps evenly.
    const int stepsPerFrame =
        static_cast<int>(std::lround(1.0 / (options.fps * demo.timestep())));
    if (stepsPerFrame < 1) {
        throw std::invalid_argument(
            "--fps is too high for the model's timestep");
    }
    const double fps = 1.0 / (stepsPerFrame * demo.timestep());
    const int frameCount = static_cast<int>(std::ceil(demo.duration() * fps));
    std::printf("recording %d frames at %.3g fps (%d steps/frame)\n",
                frameCount, fps, stepsPerFrame);

    VideoWriter video(options.outputPath, renderer.width(), renderer.height(),
                      fps);
    std::printf("ffmpeg: %s\n", video.command().c_str());

    std::string lastLabel;
    for (int frame = 0; frame < frameCount; ++frame) {
        std::string label;
        for (int step = 0; step < stepsPerFrame; ++step) {
            const Setpoint& setpoint = demo.step();
            if (step == 0) {
                label = setpoint.label;
            }
        }
        if (label != lastLabel) {
            std::printf("[%6.2f s] %s\n", demo.time(), label.c_str());
            lastLabel = label;
        }

        const auto& rgb = renderer.render(demo.data(), label);
        video.write(rgb);

        if (frame % static_cast<int>(std::lround(fps)) == 0) {
            std::printf("  frame %4d/%d  t=%.2fs  cube z=%.4f\n", frame,
                        frameCount, demo.time(), demo.cubePosition()[2]);
        }
    }

    const bool encoded = video.finish();
    std::printf("ffmpeg exit: %s\n", encoded ? "ok" : "FAILED");
    printVec("final tool", demo.toolPosition());
    printVec("final cube", demo.cubePosition());
    return encoded ? 0 : 1;
}

} // namespace pickdemo
