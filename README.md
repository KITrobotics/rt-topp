# Real-Time Time-Optimal Trajectory Parameterization

## Setup and build for users

     sudo apt install libboost-all-dev libeigen3-dev libgtest-dev libbenchmark-dev ninja-build
     git clone --recursive https://github.com/KITrobotics/rt-topp.git rttopp
     cd rttopp
     mkdir build && cd build
     cmake ../ -G Ninja
     ninja

## Additional steps for developers
     sudo apt install clang-10 clang-format-10 clang-tidy-10 dvipng texlive-latex-extra texlive-fonts-recommended cm-super texlive-science
     pip3 install -r scripts/requirements.txt
     python3 -m pre_commit install

## Examples
Files in `examples` contain some usage examples. They parametrize random paths.

`scripts` contains a script for plotting of generated trajectories.

## Tests
Tests can be run with `ctest --verbose` in the build directory.
