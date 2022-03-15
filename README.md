# Real-Time Time-Optimal Trajectory Parameterization

## Setup and build (all developer dependencies and developer setup)

     sudo apt install clang-10 clang-format-10 clang-tidy-10 libboost-all-dev libeigen3-dev libgtest-dev libbenchmark-dev ninja-build
     git clone --recursive https://github.com/KITrobotics/rttopp.git rttopp
     cd rttopp
     pip3 install -r scripts/requirements.txt
     python3 -m pre_commit install
     mkdir build && cd build
     cmake ../ -G Ninja
     ninja

## Examples
Files in `examples` contain some usage examples. They parametrize random paths.

`scripts` contains a script for plotting of generated trajectories.

## Tests
Tests can be run with `ctest --verbose` in the build directory.
