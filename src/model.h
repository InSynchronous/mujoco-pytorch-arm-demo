#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include <mujoco/mujoco.h>

namespace pickdemo {

struct ModelDeleter {
    void operator()(mjModel* model) const noexcept { mj_deleteModel(model); }
};

struct DataDeleter {
    void operator()(mjData* data) const noexcept { mj_deleteData(data); }
};

using ModelPtr = std::unique_ptr<mjModel, ModelDeleter>;
using DataPtr = std::unique_ptr<mjData, DataDeleter>;

inline ModelPtr loadModel(const std::string& path) {
    char error[1000] = {};
    ModelPtr model(mj_loadXML(path.c_str(), nullptr, error, sizeof(error)));
    if (!model) {
        throw std::runtime_error("failed to load '" + path + "': " + error);
    }
    return model;
}

inline DataPtr makeData(const mjModel& model) {
    DataPtr data(mj_makeData(&model));
    if (!data) {
        throw std::runtime_error("mj_makeData failed");
    }
    return data;
}

} // namespace pickdemo
