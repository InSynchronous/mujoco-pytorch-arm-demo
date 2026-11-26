#include "video_writer.h"

#include <cstdio>
#include <stdexcept>
#include <sys/wait.h>

namespace pickdemo {

VideoWriter::VideoWriter(std::string path, int width, int height, double fps) {
    if (width <= 0 || height <= 0 || !(fps > 0.0)) {
        throw std::invalid_argument("invalid video dimensions or frame rate");
    }

    char buffer[1024];
    std::snprintf(buffer, sizeof(buffer),
                  "ffmpeg -hide_banner -loglevel error -y -f rawvideo "
                  "-pix_fmt rgb24 -s %dx%d -r %.6g -i - -vf vflip "
                  "-c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p "
                  "-movflags +faststart \"%s\"",
                  width, height, fps, path.c_str());
    command_ = buffer;
    pipe_ = popen(command_.c_str(), "w");
    if (!pipe_) {
        throw std::runtime_error("could not start ffmpeg (is it installed?)");
    }
    frameBytes_ = static_cast<size_t>(width) * height * 3;
}

VideoWriter::~VideoWriter() {
    if (pipe_) {
        std::fflush(pipe_);
        pclose(pipe_);
    }
}

void VideoWriter::write(const std::vector<unsigned char>& rgb) {
    if (!pipe_) {
        throw std::logic_error("VideoWriter::write after finish");
    }
    if (rgb.size() != frameBytes_) {
        throw std::invalid_argument("frame size does not match the writer");
    }
    if (std::fwrite(rgb.data(), 1, rgb.size(), pipe_) != rgb.size()) {
        throw std::runtime_error("ffmpeg pipe write failed");
    }
}

bool VideoWriter::finish() {
    if (!pipe_) {
        return finished_;
    }
    std::fflush(pipe_);
    const int status = pclose(pipe_);
    pipe_ = nullptr;
    finished_ = status != -1 && WIFEXITED(status) &&
                WEXITSTATUS(status) == 0;
    return finished_;
}

} // namespace pickdemo
