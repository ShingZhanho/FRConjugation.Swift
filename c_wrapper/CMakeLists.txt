cmake_minimum_required(VERSION 3.18)
project(FRConjugation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Find LibTorch ────────────────────────────────────────────────────
# Set CMAKE_PREFIX_PATH before running cmake:
#   cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
find_package(Torch REQUIRED)

# ── nlohmann/json (header-only) ─────────────────────────────────────
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/json.hpp")
    message(STATUS "Downloading nlohmann/json single header…")
    file(DOWNLOAD
        "https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/json.hpp"
        SHOW_PROGRESS
    )
endif()

# ── Shared library ──────────────────────────────────────────────────
add_library(frconjugation SHARED conjugation.cpp)
target_include_directories(frconjugation
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(frconjugation PRIVATE ${TORCH_LIBRARIES})

# ── Static library ──────────────────────────────────────────────────
add_library(frconjugation_static STATIC conjugation.cpp)
target_include_directories(frconjugation_static
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(frconjugation_static PRIVATE ${TORCH_LIBRARIES})

# ── Test executable ─────────────────────────────────────────────────
add_executable(test_conjugation test_conjugation.cpp)
target_link_libraries(test_conjugation PRIVATE frconjugation ${TORCH_LIBRARIES})
target_include_directories(test_conjugation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# macOS rpath
if(APPLE)
    set_target_properties(frconjugation PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_RPATH   "${TORCH_INSTALL_PREFIX}/lib"
    )
    set_target_properties(test_conjugation PROPERTIES
        BUILD_RPATH   "${TORCH_INSTALL_PREFIX}/lib"
    )
endif()

message(STATUS "LibTorch: ${TORCH_INSTALL_PREFIX}")
