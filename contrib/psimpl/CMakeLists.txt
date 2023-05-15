cmake_minimum_required(VERSION 3.1)

project(psimpl LANGUAGES CXX)

message("Creating psimpl")

add_library(psimpl INTERFACE)
SET(CMAKE_CXX_STANDARD 11)

target_include_directories(psimpl INTERFACE .)
