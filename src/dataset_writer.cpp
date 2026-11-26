#include "dataset_writer.h"

#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <utility>

namespace pickdemo {

DatasetWriter::DatasetWriter(std::string directory, int width, int height)
    : directory_(std::move(directory)), width_(width), height_(height) {
    if (width_ <= 0 || height_ <= 0) {
        throw std::invalid_argument("invalid frame size");
    }
    std::filesystem::create_directories(directory_);
    const std::string base = directory_ + "/";
    frames_ = std::fopen((base + "frames.bin").c_str(), "wb");
    steps_ = std::fopen((base + "steps.csv").c_str(), "wb");
    episodes_ = std::fopen((base + "episodes.csv").c_str(), "wb");
    if (!frames_ || !steps_ || !episodes_) {
        throw std::runtime_error("could not open dataset files in " +
                                 directory_);
    }
    std::fprintf(steps_,
                 "episode,frame,q0,q1,q2,q3,q4,q5,"
                 "qd0,qd1,qd2,qd3,qd4,qd5,"
                 "a0,a1,a2,a3,a4,a5,cube_x,cube_y,label\n");
    std::fprintf(episodes_, "episode,cube_x,cube_y,success\n");
    flipped_.resize(static_cast<size_t>(width_) * height_ * 3);
}

DatasetWriter::~DatasetWriter() { finish(); }

void DatasetWriter::beginEpisode(int episode) {
    episode_ = episode;
    frame_ = 0;
}

void DatasetWriter::addFrame(const std::vector<unsigned char>& rgb,
                             const Transition& transition) {
    if (!frames_) {
        throw std::logic_error("DatasetWriter is closed");
    }
    if (rgb.size() != flipped_.size()) {
        throw std::invalid_argument("frame size does not match the dataset");
    }

    // OpenGL readback is bottom-up; store top-down for image loaders.
    const size_t rowBytes = static_cast<size_t>(width_) * 3;
    for (int y = 0; y < height_; ++y) {
        const unsigned char* source =
            rgb.data() + static_cast<size_t>(height_ - 1 - y) * rowBytes;
        std::memcpy(flipped_.data() + static_cast<size_t>(y) * rowBytes,
                    source, rowBytes);
    }
    std::fwrite(flipped_.data(), 1, flipped_.size(), frames_);

    std::fprintf(
        steps_,
        "%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,"
        "%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%s\n",
        episode_, frame_, transition.q[0], transition.q[1], transition.q[2],
        transition.q[3], transition.q[4], transition.q[5],
        transition.qdot[0], transition.qdot[1], transition.qdot[2],
        transition.qdot[3], transition.qdot[4], transition.qdot[5],
        transition.action[0], transition.action[1], transition.action[2],
        transition.action[3], transition.action[4], transition.action[5],
        transition.cube[0], transition.cube[1], transition.label.c_str());
    ++frame_;
}

void DatasetWriter::endEpisode(bool success,
                               const std::array<double, 2>& finalCube) {
    if (!episodes_) {
        return;
    }
    std::fprintf(episodes_, "%d,%.9g,%.9g,%d\n", episode_, finalCube[0],
                 finalCube[1], success ? 1 : 0);
    std::fflush(episodes_);
    std::fflush(steps_);
    std::fflush(frames_);
}

void DatasetWriter::finish() {
    if (frames_) {
        std::fclose(frames_);
        frames_ = nullptr;
    }
    if (steps_) {
        std::fclose(steps_);
        steps_ = nullptr;
    }
    if (episodes_) {
        std::fclose(episodes_);
        episodes_ = nullptr;
    }
}

void DatasetWriter::writeInfo(const std::string& directory,
                              const DatasetInfo& info) {
    std::filesystem::create_directories(directory);
    const std::string path = directory + "/info.json";
    FILE* file = std::fopen(path.c_str(), "wb");
    if (!file) {
        throw std::runtime_error("could not write " + path);
    }
    std::fprintf(file, "{\n");
    std::fprintf(file, "  \"width\": %d,\n  \"height\": %d,\n", info.width,
                 info.height);
    std::fprintf(file,
                 "  \"control_dt\": %.9g,\n  \"sim_dt\": %.9g,\n"
                 "  \"duration\": %.9g,\n  \"ticks\": %d,\n",
                 info.controlDt, info.simDt, info.duration, info.ticks);
    std::fprintf(file, "  \"steps_per_tick\": %d,\n", info.stepsPerTick);
    std::fprintf(file,
                 "  \"home_qpos\": [%.9g, %.9g, %.9g, %.9g, %.9g, %.9g],\n",
                 info.homeQpos[0], info.homeQpos[1], info.homeQpos[2],
                 info.homeQpos[3], info.homeQpos[4], info.homeQpos[5]);
    std::fprintf(file, "  \"home_position\": [%.9g, %.9g, %.9g],\n",
                 info.homePosition[0], info.homePosition[1],
                 info.homePosition[2]);
    std::fprintf(file, "  \"place_position\": [%.9g, %.9g, %.9g],\n",
                 info.placePosition[0], info.placePosition[1],
                 info.placePosition[2]);
    std::fprintf(file, "  \"arm_gains\": [%.9g, %.9g],\n", info.armGains.kp,
                 info.armGains.kv);
    std::fprintf(file, "  \"gripper_gains\": [%.9g, %.9g],\n",
                 info.gripperGains.kp, info.gripperGains.kv);
    std::fprintf(file, "  \"gripper_open\": %.9g,\n  \"gripper_closed\": %.9g,\n",
                 info.gripperTravel.open, info.gripperTravel.closed);
    std::fprintf(file,
                 "  \"camera\": {\"azimuth\": %.9g, \"elevation\": %.9g, "
                 "\"distance\": %.9g, \"lookat\": [%.9g, %.9g, %.9g]},\n",
                 info.camera.azimuth, info.camera.elevation, info.camera.distance,
                 info.camera.lookat[0], info.camera.lookat[1],
                 info.camera.lookat[2]);
    std::fprintf(file,
                 "  \"shard\": %d, \"shards\": %d, \"episode_start\": %d, "
                 "\"episode_count\": %d, \"seed\": %u,\n",
                 info.shardIndex, info.shardCount, info.episodeStart,
                 info.episodeCount, info.seed);
    std::fprintf(file,
                 "  \"cube_bounds\": [%.9g, %.9g, %.9g, %.9g],\n",
                 info.cubeXMin, info.cubeXMax, info.cubeYMin, info.cubeYMax);
    std::fprintf(file, "  \"cube_radius\": [%.9g, %.9g],\n", info.minRadius,
                 info.maxRadius);
    std::fprintf(file, "  \"place_clearance\": %.9g\n", info.placeClearance);
    std::fprintf(file, "}\n");
    std::fclose(file);
}

} // namespace pickdemo
