cmake_minimum_required(VERSION 3.16)

project(demo-yolox)

# demo-yolox — detects objects in one image and reports timings
add_executable(${PROJECT_NAME}
    demo.cpp
    processing.cpp
)

add_dependencies(${PROJECT_NAME} yolox-shaders)
