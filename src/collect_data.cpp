// Generates randomized expert demonstrations for behavior cloning.
//
//   collect_data --out data --episodes 2000 --shard 0 --shards 6 --seed 1
//
// Each shard owns a deterministic subset of the episode indices, so shards can
// run in parallel and merge by concatenation.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "dataset_writer.h"
#include "episode.h"
#include "renderer.h"

namespace {

struct Options {
    std::string out = "data";
    int episodes = 100;
    int shard = 0;
    int shards = 1;
    int width = 128;
    int height = 128;
    double fps = 10.0;
    unsigned seed = 1;
    pickdemo::CameraConfig camera;
};

int parseInt(const char* text) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid integer '") + text +
                                    "'");
    }
    return static_cast<int>(value);
}

double parseDouble(const char* text) {
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid number '") + text +
                                    "'");
    }
    return value;
}

Options parseArgs(int argc, char** argv) {
    // Default view: close enough that the 4 cm cube is ~10 pixels at 128.
    Options options;
    options.camera = {-45.0, 0.95, -35.0, {0.31, 0.00, 0.11}};

    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                throw std::invalid_argument("missing value for " + flag);
            }
            return argv[++i];
        };
        if (flag == "--out") {
            options.out = next();
        } else if (flag == "--episodes") {
            options.episodes = parseInt(next());
        } else if (flag == "--shard") {
            options.shard = parseInt(next());
        } else if (flag == "--shards") {
            options.shards = parseInt(next());
        } else if (flag == "--width") {
            options.width = parseInt(next());
        } else if (flag == "--height") {
            options.height = parseInt(next());
        } else if (flag == "--fps") {
            options.fps = parseDouble(next());
        } else if (flag == "--seed") {
            options.seed = static_cast<unsigned>(parseInt(next()));
        } else if (flag == "--az") {
            options.camera.azimuth = parseDouble(next());
        } else if (flag == "--elev") {
            options.camera.elevation = parseDouble(next());
        } else if (flag == "--dist") {
            options.camera.distance = parseDouble(next());
        } else if (flag == "--lookx") {
            options.camera.lookat[0] = parseDouble(next());
        } else if (flag == "--looky") {
            options.camera.lookat[1] = parseDouble(next());
        } else if (flag == "--lookz") {
            options.camera.lookat[2] = parseDouble(next());
        } else {
            throw std::invalid_argument("unknown option '" + flag + "'");
        }
    }

    if (options.episodes < 1 || options.shards < 1 || options.shard < 0 ||
        options.shard >= options.shards) {
        throw std::invalid_argument("invalid episode/shard counts");
    }
    if (options.width < 16 || options.height < 16 || !(options.fps > 0.0)) {
        throw std::invalid_argument("invalid size or frame rate");
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    using namespace pickdemo;

    Options options;
    try {
        options = parseArgs(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }

    const int base = options.episodes / options.shards;
    const int remainder = options.episodes % options.shards;
    const int episodeStart =
        options.shard * base + std::min(options.shard, remainder);
    const int episodeCount = base + (options.shard < remainder ? 1 : 0);

    char shardName[32];
    std::snprintf(shardName, sizeof(shardName), "shard_%03d", options.shard);
    const std::string directory = options.out + "/" + shardName;

    try {
        EpisodeConfig config;
        config.controlDt = 1.0 / options.fps;
        config.seed = options.seed;

        PickPlaceEpisode episode("scene.xml", config);
        Renderer renderer(episode.model(), options.width, options.height, "",
                          options.camera);
        DatasetWriter writer(directory, options.width, options.height);

        DatasetInfo info;
        info.width = options.width;
        info.height = options.height;
        info.controlDt = episode.controlDt();
        info.simDt = episode.simDt();
        info.duration = episode.ticks() * episode.controlDt();
        info.ticks = episode.ticks();
        info.stepsPerTick =
            static_cast<int>(std::lround(episode.controlDt() / episode.simDt()));
        info.homeQpos = episode.homeQpos();
        info.homePosition = config.path.home;
        info.placePosition = config.path.place;
        info.armGains = config.armGains;
        info.gripperGains = config.gripperGains;
        info.gripperTravel = config.gripperTravel;
        info.camera = options.camera;
        info.shardIndex = options.shard;
        info.shardCount = options.shards;
        info.episodeStart = episodeStart;
        info.episodeCount = episodeCount;
        info.seed = options.seed;
        info.cubeXMin = config.cubeXMin;
        info.cubeXMax = config.cubeXMax;
        info.cubeYMin = config.cubeYMin;
        info.cubeYMax = config.cubeYMax;
        info.minRadius = config.minRadius;
        info.maxRadius = config.maxRadius;
        info.placeClearance = config.placeClearance;
        DatasetWriter::writeInfo(directory, info);

        std::printf("shard %d/%d: episodes %d..%d (%d) -> %s\n",
                    options.shard, options.shards, episodeStart,
                    episodeStart + episodeCount - 1, episodeCount,
                    directory.c_str());
        std::fflush(stdout);

        int successes = 0;
        const auto start = std::chrono::steady_clock::now();
        for (int local = 0; local < episodeCount; ++local) {
            const int episodeIndex = episodeStart + local;
            episode.reset(static_cast<std::uint32_t>(episodeIndex));
            writer.beginEpisode(episodeIndex);

            const int ticks = episode.ticks();
            for (int tick = 0; tick < ticks; ++tick) {
                // The frame must show the state before the action is applied.
                const auto& rgb = renderer.render(episode.data(), "");
                const Transition& transition = episode.step();
                writer.addFrame(rgb, transition);
            }

            const bool ok = episode.succeeded();
            successes += ok ? 1 : 0;
            writer.endEpisode(ok, episode.cubePosition());

            if ((local + 1) % 20 == 0 || local + 1 == episodeCount) {
                const auto elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start);
                std::printf("  %4d/%d episodes  (%.2f s/episode, %d ok)\n",
                            local + 1, episodeCount,
                            elapsed.count() / (local + 1), successes);
                std::fflush(stdout);
            }
        }

        std::printf("done: %d/%d episodes placed the cube\n", successes,
                    episodeCount);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    return 0;
}
