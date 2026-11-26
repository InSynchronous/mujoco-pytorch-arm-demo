#pragma once

#include <iosfwd>
#include <string>

#include "renderer.h"

namespace pickdemo {

struct Options {
    std::string modelPath = "scene.xml";
    std::string outputPath = "pick_and_place.mp4";
    std::string title = "MuJoCo  |  4-DOF arm, IK pick & place";

    std::string stillPath;    // if set, write one PPM instead of a video
    double stillTime = -1.0;  // < 0: the middle of the trajectory

    double fps = 50.0;
    int width = 1920;
    int height = 1080;

    CameraConfig camera;
    bool help = false;
};

// Throws std::invalid_argument on malformed flags.
Options parseOptions(int argc, char** argv);
void printUsage(std::ostream& out, const char* program);

} // namespace pickdemo
