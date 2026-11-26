#include "options.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <ostream>
#include <stdexcept>
#include <string>

namespace pickdemo {
namespace {

const char* takeValue(int argc, char** argv, int& index, const char* flag) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + flag);
    }
    return argv[++index];
}

double parseDouble(const char* text, const char* flag) {
    char* end = nullptr;
    errno = 0;
    const double value = std::strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid number for ") + flag +
                                    ": '" + text + "'");
    }
    return value;
}

int parseInt(const char* text, const char* flag) {
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 0 ||
        value > 1000000) {
        throw std::invalid_argument(std::string("invalid value for ") + flag +
                                    ": '" + text + "'");
    }
    return static_cast<int>(value);
}

bool matches(const char* argument, const char* shortName,
             const char* longName) {
    return std::strcmp(argument, shortName) == 0 ||
           std::strcmp(argument, longName) == 0;
}

} // namespace

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const char* flag = argv[i];
        if (matches(flag, "-h", "--help")) {
            options.help = true;
        } else if (std::strcmp(flag, "--model") == 0) {
            options.modelPath = takeValue(argc, argv, i, flag);
        } else if (std::strcmp(flag, "--out") == 0) {
            options.outputPath = takeValue(argc, argv, i, flag);
        } else if (std::strcmp(flag, "--title") == 0) {
            options.title = takeValue(argc, argv, i, flag);
        } else if (std::strcmp(flag, "--still") == 0) {
            options.stillPath = takeValue(argc, argv, i, flag);
        } else if (std::strcmp(flag, "--time") == 0) {
            options.stillTime = parseDouble(takeValue(argc, argv, i, flag), flag);
        } else if (std::strcmp(flag, "--fps") == 0) {
            options.fps = parseDouble(takeValue(argc, argv, i, flag), flag);
        } else if (std::strcmp(flag, "--width") == 0) {
            options.width = parseInt(takeValue(argc, argv, i, flag), flag);
        } else if (std::strcmp(flag, "--height") == 0) {
            options.height = parseInt(takeValue(argc, argv, i, flag), flag);
        } else if (std::strcmp(flag, "--az") == 0) {
            options.camera.azimuth =
                parseDouble(takeValue(argc, argv, i, flag), flag);
        } else if (std::strcmp(flag, "--elev") == 0) {
            options.camera.elevation =
                parseDouble(takeValue(argc, argv, i, flag), flag);
        } else if (std::strcmp(flag, "--dist") == 0) {
            options.camera.distance =
                parseDouble(takeValue(argc, argv, i, flag), flag);
        } else if (std::strcmp(flag, "--lookx") == 0) {
            options.camera.lookat[0] =
                parseDouble(takeValue(argc, argv, i, flag), flag);
        } else if (std::strcmp(flag, "--looky") == 0) {
            options.camera.lookat[1] =
                parseDouble(takeValue(argc, argv, i, flag), flag);
        } else if (std::strcmp(flag, "--lookz") == 0) {
            options.camera.lookat[2] =
                parseDouble(takeValue(argc, argv, i, flag), flag);
        } else {
            throw std::invalid_argument(std::string("unknown option '") + flag +
                                        "' (try --help)");
        }
    }

    if (options.width < 16 || options.height < 16) {
        throw std::invalid_argument("--width/--height must be at least 16");
    }
    if (!(options.fps > 0.0)) {
        throw std::invalid_argument("--fps must be positive");
    }
    return options;
}

void printUsage(std::ostream& out, const char* program) {
    out << "Usage: " << program << " [options]\n\n"
        << "Records a MuJoCo pick-and-place demo to an MP4 by default.\n\n"
        << "  --model FILE      MuJoCo scene            (scene.xml)\n"
        << "  --out FILE        output video            (pick_and_place.mp4)\n"
        << "  --still FILE      write one PPM frame instead of a video\n"
        << "  --time SECONDS    time for --still        (middle of motion)\n"
        << "  --fps N           video frame rate        (50)\n"
        << "  --width N         frame width             (1920)\n"
        << "  --height N        frame height            (1080)\n"
        << "  --title TEXT      overlay title\n"
        << "  --az DEG          camera azimuth          (-45)\n"
        << "  --elev DEG        camera elevation        (-21)\n"
        << "  --dist M          camera distance         (1.28)\n"
        << "  --lookx/y/z M     camera look-at point    (0.255 -0.04 0.19)\n"
        << "  -h, --help        show this message\n";
}

} // namespace pickdemo
