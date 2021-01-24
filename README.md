# PyPnC
PyPnC is a python library designed for generating trajectories for a robot
system and stabilizing the system over the trajectories.

## Installation
- Install [Conda](https://docs.anaconda.com/anaconda/install/)
- Clone the repository
```
$ git clone https://github.com/junhyeokahn/PyPnC.git
```
- Create a virtual environment and install Dependancies
```
$ conda env create -f environment.yml
```
- Activate the environment
```
$ conda activate ASE389
```

## Running Examples
### Three Link Manipulator Control with Operational Space Control
- Run
```
$ python simulator/pybullet/manipulator_main.py
```
### Atlas Walking Control with DCM planning and IHWBC
- Run
```
$ python simulator/pybullet/atlas_dynamics_main.py
```
- Send Walking Commands through Keystroke Interface. For example, hit 8 for forward walking, hit 5 for in-place walking, hit 4 for leftward walking, hit 6 for rightward walking, hit 2 for backward walking, hit 7 for ccw turning, and hit 9 for cw turning.
- Plot the Results
```
$ python plot/atlas/plot_task.py --file=data/history.pkl
```
### TOWR+
- Compile and Run the Code
```
$ mkdir build && cd build && cmake .. && make -j6
$ ./atlas_two_step_yaml_test
```
- Plot the Optimized Trajectory
```
$ python plot/plot_towr_plus_trajectory.py --file=data/atlas_two_step_yaml_test.yaml
```
- Visualize the Optimized Trajectory with Atlas
```
$ python simulator/pybullet/atlas_kinematics_main.py --file=data/atlas_two_step_yaml_test.yaml
```
