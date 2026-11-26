cmake_minimum_required(VERSION 3.16)
project(mujoco_pick_demo VERSION 1.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

# MuJoCo is built from the pinned source submodule in third_party/mujoco.
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mujoco/CMakeLists.txt")
  message(FATAL_ERROR
    "MuJoCo submodule is missing. Run: git submodule update --init --recursive")
endif()
set(MUJOCO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MUJOCO_BUILD_SIMULATE OFF CACHE BOOL "" FORCE)
set(MUJOCO_BUILD_STUDIO OFF CACHE BOOL "" FORCE)
set(MUJOCO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MUJOCO_TEST_PYTHON_UTIL OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/mujoco EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# core: model loading, trajectory, IK and control. No window or encoder code.
# ---------------------------------------------------------------------------
add_library(pickdemo_core STATIC
  src/demo.cpp
  src/episode.cpp
  src/ik.cpp
  src/robot.cpp
  src/trajectory.cpp)
target_include_directories(pickdemo_core PUBLIC src)
target_link_libraries(pickdemo_core PUBLIC mujoco::mujoco)

# ---------------------------------------------------------------------------
# io: renderer, video writer, CLI and app wiring.
# ---------------------------------------------------------------------------
add_library(pickdemo_io STATIC
  src/app.cpp
  src/dataset_writer.cpp
  src/options.cpp
  src/renderer.cpp
  src/video_writer.cpp)
target_link_libraries(pickdemo_io PUBLIC pickdemo_core glfw OpenGL::GL)

add_executable(pick_demo src/main.cpp)
target_link_libraries(pick_demo PRIVATE pickdemo_io)

add_executable(collect_data src/collect_data.cpp)
target_link_libraries(collect_data PRIVATE pickdemo_io)

foreach(target pickdemo_core pickdemo_io pick_demo collect_data)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
endforeach()

# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
include(CTest)
if(BUILD_TESTING)
  add_executable(core_tests tests/core_tests.cpp)
  target_link_libraries(core_tests PRIVATE pickdemo_core)
  target_compile_options(core_tests PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME core_tests
           COMMAND core_tests "${CMAKE_CURRENT_SOURCE_DIR}/scene.xml")
endif()
