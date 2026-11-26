#pragma once

#include <array>
#include <string>
#include <string_view>
#include <vector>

#include <mujoco/mujoco.h>

struct GLFWwindow;

namespace pickdemo {

struct CameraConfig {
    double azimuth = -45.0;
    double distance = 1.28;
    double elevation = -21.0;
    std::array<double, 3> lookat{0.255, -0.04, 0.19};
};

// Offscreen MuJoCo renderer. Opens a hidden GLFW window, owns the mjrContext
// and mjvScene, and reads back RGB24 frames. The model must outlive it.
class Renderer {
  public:
    Renderer(const mjModel& model, int width, int height, std::string title,
             CameraConfig camera = {});
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    int width() const { return width_; }
    int height() const { return height_; }

    // Draws the current state plus overlays. The returned buffer is RGB24,
    // bottom-up (OpenGL order), width * height * 3 bytes.
    const std::vector<unsigned char>& render(mjData& data,
                                             std::string_view label);

    void setCamera(const CameraConfig& camera);

    // Writes bottom-up RGB24 data as a binary PPM, flipped to top-down.
    static void writePpm(const std::string& path,
                         const std::vector<unsigned char>& rgb, int width,
                         int height);

  private:
    const mjModel* model_ = nullptr;
    GLFWwindow* window_ = nullptr;
    mjvCamera camera_{};
    mjvOption options_{};
    mjvScene scene_{};
    mjrContext context_{};
    mjrRect viewport_{0, 0, 0, 0};
    std::string title_;
    std::vector<unsigned char> pixels_;
    int width_ = 0;
    int height_ = 0;
};

} // namespace pickdemo
