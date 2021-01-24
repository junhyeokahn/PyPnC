# PyPnC
PyPnC is a python library designed for generating trajectories for a robot
system and stabilizing the system over the trajectories.

## Installation
- Install [conda](https://docs.anaconda.com/anaconda/install/)
- Clone the repository ```$ git clone https://github.com/junhyeokahn/PyPnC.git```
- Create a virtual environment and install dependancies ```$ conda env create -f environment.yml```
- Activate the environment ```$ conda activate PyPnC```

## Running Examples
### Three Link Manipulator Control with Operational Space Control
- Run ```$ python simulator/pybullet/manipulator_main.py```
### Atlas Walking Control with DCM planning and IHWBC
- Run ```$ python simulator/pybullet/atlas_dynamics_main.py```
- Send walking commands through keystroke interface. For example, press ```8``` for forward walking, press ```5``` for in-place walking, press ```4``` for leftward walking, press ```6``` for rightward walking, press ```2``` for backward walking, press ```7``` for ccw turning, and press ```9``` for cw turning.
- Plot the results
```
$ python plot/atlas/plot_task.py --file=data/history.pkl
```
### TOWR+
- Compile and run the code ```$ mkdir build && cd build && cmake .. && make -j6 && ./atlas_two_step_yaml_test```
- Plot the optimized trajectory ```$ python plot/plot_towr_plus_trajectory.py --file=data/atlas_two_step_yaml_test.yaml```
- Visualize the optimized trajectory with Atlas ```$ python simulator/pybullet/atlas_kinematics_main.py --file=data/atlas_two_step_yaml_test.yaml```
