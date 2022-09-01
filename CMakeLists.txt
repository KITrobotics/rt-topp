cmake_minimum_required(VERSION 3.1)
project(rttopp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE AND CMAKE_BUILD_TYPE MATCHES Release)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

set(JSON_BuildTests
    OFF
    CACHE INTERNAL "")
add_subdirectory(external/json)

find_package(benchmark REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)

set(GTest_ROOT /usr/src/googletest/googletest)
add_subdirectory(${GTest_ROOT} "${CMAKE_CURRENT_BINARY_DIR}/googletest"
                 EXCLUDE_FROM_ALL)
enable_testing()

add_compile_options(
  -Wall
  -Wextra
  -fdiagnostics-color=always
  -Wpedantic
  -Wredundant-decls
  -Wcast-qual
  -Wfloat-equal
  -Wshadow)

add_library(${PROJECT_NAME} src/demo_trajectories.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen
                                             nlohmann_json::nlohmann_json)

add_executable(param_random_waypoints_example_generic
               examples/param_random_waypoints_generic.cpp)
target_link_libraries(param_random_waypoints_example_generic
                      PRIVATE ${PROJECT_NAME})
add_executable(param_random_waypoints_example_iiwa
               examples/param_random_waypoints_iiwa.cpp)
target_link_libraries(param_random_waypoints_example_iiwa
                      PRIVATE ${PROJECT_NAME})
add_executable(param_random_waypoints_generic_sampling_example
               examples/param_random_waypoints_generic_sampling.cpp)
target_link_libraries(param_random_waypoints_generic_sampling_example
                      PRIVATE ${PROJECT_NAME})
add_executable(param_failed_random_waypoints_example_generic
               examples/param_failed_random_waypoints_generic.cpp)
target_link_libraries(param_failed_random_waypoints_example_generic
                      PRIVATE ${PROJECT_NAME} stdc++fs)

add_executable(path_velocity_limit_tests tests/path_velocity_limit.cpp)
target_link_libraries(path_velocity_limit_tests PRIVATE ${PROJECT_NAME}
                                                        gtest_main)
add_test(NAME path_velocity_limit_tests COMMAND path_velocity_limit_tests)

add_executable(preprocessing_tests tests/test_preprocessing.cpp)
target_link_libraries(preprocessing_tests PRIVATE ${PROJECT_NAME} gtest_main)
add_test(NAME preprocessing_tests COMMAND preprocessing_tests)

add_executable(param_tests tests/param.cpp)
target_link_libraries(param_tests PRIVATE ${PROJECT_NAME} gtest_main)
add_test(NAME param_tests COMMAND param_tests)
