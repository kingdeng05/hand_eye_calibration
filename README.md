# Kinetic System calibration 

## Introduction 
This repo hosts the toolchains to calibrate Kinetic system, including feature extraction, backend optimization, etc.

## Design 
[Design Doc](https://kinetic-automation.atlassian.net/wiki/spaces/~71202021bf84a854e648988072ebeae026b931/pages/143523963/Graph+optimization+for+joint+calibration)

## Develop
### Install GTSAM
GTSAM is the lib developed by Georgia Tech Institute for factor graph optimization. It's a proven all around solution for SLAM applications.
1. clone [gtsam4](https://github.com/borglab/gtsam.git) to a folder and check out to commit id: 6425000775da29c93719f37dce3f2de38a0064ec (4.0.3)
2. Install by
   ```
   cd gtsam && mkdir build && cd build && cmake .. && sudo make install
   ```

### Install the repo
```
bash scripts/install.sh
```

## Folder structure
1. All the cpp and pybind files lie in kinetic_backend folder.
2. All the system calibration files lie in system_calibration folder.
    1. `apps` are where calibration interface, where you actually generate calibration results from it.
    2. `system_calibration` hosts the source files for factor graphs, frontend feature extraction, etc.
