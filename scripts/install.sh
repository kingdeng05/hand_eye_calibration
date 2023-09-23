#!/bin/bash

# Exit script on first error
set -e

# Define the parent directory
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PARENT_DIR=$(realpath "${SCRIPT_DIR}/..")

# Navigate to A, create a build directory, run cmake, and install
pushd "${PARENT_DIR}/kinetic_backend"
mkdir -p build
pushd build
cmake ..
make
sudo make install
popd  # Return from build directory to A
popd  # Return from A to PARENT_DIR

# Navigate to B and install Python project
pushd "${PARENT_DIR}/system_calibration"
pip3 install -e .
popd