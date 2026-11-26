#pragma once

#include <cstdio>
#include <string>
#include <vector>

namespace pickdemo {

// Streams raw RGB24 frames into an ffmpeg process (H.264 MP4). The frame
// format is fixed at construction; every write() must match its size.
class VideoWriter {
  public:
    VideoWriter(std::string path, int width, int height, double fps);
    ~VideoWriter();

    VideoWriter(const VideoWriter&) = delete;
    VideoWriter& operator=(const VideoWriter&) = delete;

    void write(const std::vector<unsigned char>& rgb);

    // Flushes and waits for ffmpeg. Returns true if it exited successfully.
    bool finish();

    const std::string& command() const { return command_; }

  private:
    std::string command_;
    FILE* pipe_ = nullptr;
    size_t frameBytes_ = 0;
    bool finished_ = false;
};

} // namespace pickdemo
