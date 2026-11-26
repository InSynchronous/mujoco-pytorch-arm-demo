#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "episode.h"
#include "renderer.h"
#include "robot.h"

namespace pickdemo {

struct DatasetInfo {
    int width = 128;
    int height = 128;
    double controlDt = 0.1;
    double simDt = 0.002;
    double duration = 0.0;
    int ticks = 0;
    int stepsPerTick = 50;
    std::array<double, kMotorCount> homeQpos{};
    std::array<double, 3> homePosition{};
    std::array<double, 3> placePosition{};
    ServoGains armGains{};
    ServoGains gripperGains{};
    GripperTravel gripperTravel{};
    CameraConfig camera;

    int shardIndex = 0;
    int shardCount = 1;
    int episodeStart = 0;
    int episodeCount = 0;
    std::uint32_t seed = 1;

    double cubeXMin = 0.0, cubeXMax = 0.0;
    double cubeYMin = 0.0, cubeYMax = 0.0;
    double minRadius = 0.0, maxRadius = 0.0;
    double placeClearance = 0.0;
};

// Streams one shard of raw RGB frames and a CSV of states/actions. The frames
// are stored top-down (row 0 = top) as width*height*3 bytes each.
class DatasetWriter {
  public:
    DatasetWriter(std::string directory, int width, int height);
    ~DatasetWriter();

    DatasetWriter(const DatasetWriter&) = delete;
    DatasetWriter& operator=(const DatasetWriter&) = delete;

    void beginEpisode(int episode);
    void addFrame(const std::vector<unsigned char>& rgb,
                  const Transition& transition);
    void endEpisode(bool success, const std::array<double, 2>& finalCube);
    void finish();

    static void writeInfo(const std::string& directory,
                          const DatasetInfo& info);

  private:
    std::string directory_;
    FILE* frames_ = nullptr;
    FILE* steps_ = nullptr;
    FILE* episodes_ = nullptr;
    std::vector<unsigned char> flipped_;
    int width_ = 0;
    int height_ = 0;
    int episode_ = -1;
    int frame_ = 0;
};

} // namespace pickdemo
