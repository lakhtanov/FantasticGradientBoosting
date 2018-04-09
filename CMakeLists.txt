cmake_minimum_required(VERSION 3.8)
project(FantasticGradientBoosting)

set(CMAKE_CXX_STANDARD 17)

SET(BASEPATH "${CMAKE_SOURCE_DIR}")
INCLUDE_DIRECTORIES("${BASEPATH}")
set(SOURCE_FILES gradient_boosting_app.cpp utils/data_containers/ElementContainer.cpp utils/data_containers/ElementContainer.h)
add_executable(FantasticGradientBoosting ${SOURCE_FILES})
