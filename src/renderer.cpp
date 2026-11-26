#include "renderer.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cstdio>
#include <stdexcept>
#include <utility>

namespace pickdemo {
namespace {

int gGlfwUsers = 0;

} // namespace

Renderer::Renderer(const mjModel& model, int width, int height,
                   std::string title, CameraConfig camera)
    : model_(&model), title_(std::move(title)), width_(width),
      height_(height) {
    if (width < 1 || height < 1) {
        throw std::invalid_argument("renderer size must be positive");
    }

    if (gGlfwUsers == 0 && !glfwInit()) {
        throw std::runtime_error("GLFW initialization failed");
    }
    ++gGlfwUsers;

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    window_ = glfwCreateWindow(width, height, title_.c_str(), nullptr, nullptr);
    if (!window_) {
        if (--gGlfwUsers == 0) {
            glfwTerminate();
        }
        throw std::runtime_error("could not create an OpenGL window");
    }
    glfwMakeContextCurrent(window_);
    glfwGetFramebufferSize(window_, &width_, &height_);

    mjv_defaultCamera(&camera_);
    mjv_defaultOption(&options_);
    mjv_defaultScene(&scene_);
    mjr_defaultContext(&context_);
    mjv_makeScene(model_, &scene_, 4000);
    mjr_makeContext(model_, &context_, mjFONTSCALE_150);

    options_.sitegroup[3] = 0; // hide the tool-site marker
    options_.flags[mjVIS_CONTACTPOINT] = 0;
    options_.flags[mjVIS_CONTACTFORCE] = 0;

    setCamera(camera);
    viewport_ = {0, 0, width_, height_};
    pixels_.assign(static_cast<size_t>(width_) * height_ * 3, 0);
}

Renderer::~Renderer() {
    if (!window_) {
        return;
    }
    mjr_freeContext(&context_);
    mjv_freeScene(&scene_);
    glfwDestroyWindow(window_);
    if (--gGlfwUsers == 0) {
        glfwTerminate();
    }
}

void Renderer::setCamera(const CameraConfig& camera) {
    camera_.type = mjCAMERA_FREE;
    camera_.lookat[0] = camera.lookat[0];
    camera_.lookat[1] = camera.lookat[1];
    camera_.lookat[2] = camera.lookat[2];
    camera_.distance = camera.distance;
    camera_.azimuth = camera.azimuth;
    camera_.elevation = camera.elevation;
}

const std::vector<unsigned char>& Renderer::render(mjData& data,
                                                   std::string_view label) {
    glfwMakeContextCurrent(window_);
    mjv_updateScene(model_, &data, &options_, nullptr, &camera_, mjCAT_ALL,
                    &scene_);
    mjr_render(viewport_, &scene_, &context_);

    if (!title_.empty()) {
        mjr_overlay(mjFONT_BIG, mjGRID_TOPLEFT, viewport_, title_.c_str(),
                    nullptr, &context_);
    }
    if (!label.empty()) {
        const std::string text(label);
        mjr_overlay(mjFONT_NORMAL, mjGRID_TOPRIGHT, viewport_, text.c_str(),
                    nullptr, &context_);
    }

    mjr_readPixels(pixels_.data(), nullptr, viewport_, &context_);
    return pixels_;
}

void Renderer::writePpm(const std::string& path,
                        const std::vector<unsigned char>& rgb, int width,
                        int height) {
    FILE* file = std::fopen(path.c_str(), "wb");
    if (!file) {
        throw std::runtime_error("could not open '" + path + "' for writing");
    }
    std::fprintf(file, "P6\n%d %d\n255\n", width, height);
    for (int y = height - 1; y >= 0; --y) {
        std::fwrite(rgb.data() + static_cast<size_t>(y) * width * 3, 1,
                    static_cast<size_t>(width) * 3, file);
    }
    std::fclose(file);
}

} // namespace pickdemo
