cmake_minimum_required(VERSION 3.16)

project(demo-mnist)

# demo-mnist — classifies a digit image with timings
add_executable(${PROJECT_NAME}
    main.cpp
)
