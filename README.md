# PyPnC
PyPnC is a python library designed for generating trajectories for a robot
system and stabilizing the system over the trajectories.

## Installation
- Install [anaconda](https://docs.anaconda.com/anaconda/install/)
- Clone the repository:<br/>
```$ git clone https://github.com/junhyeokahn/PyPnC.git```
- Create a virtual environment and install dependancies:<br/>
```$ conda env create -f pypnc.yml```
- Activate the environment:<br/>
```$ conda activate pypnc```

## Running Examples
### Three Link Manipulator Control with Operational Space Control
- Run the code:<br/>
```$ python simulator/pybullet/manipulator_main.py```
### Atlas Walking Control with DCM planning and IHWBC
- Run the code:<br/>
```$ python simulator/pybullet/atlas_dynamics_main.py```
- Send walking commands through keystroke interface. For example, press ```8``` for forward walking, press ```5``` for in-place walking, press ```4``` for leftward walking, press ```6``` for rightward walking, press ```2``` for backward walking, press ```7``` for ccw turning, and press ```9``` for cw turning.
- Plot the results:<br/>
```$ python plot/atlas/plot_task.py --file=data/history.pkl```
### Atlas Locomotion Planning with TOWR+
- For TOWR+, install additional dependancy [ifopt](https://github.com/ethz-adrl/ifopt)
- Train a Composite Rigid Body Inertia network and generate files for optimization:<br/>
```$ python simulator/pybullet/atlas_crbi_trainer.py``` and press ```5``` for training
- Run ```TOWR+```:<br/>
```$ mkdir build && cd build && cmake .. && make -j6 && ./atlas_forward_walk```
- Plot the optimized trajectory:<br/>
```$ python plot/plot_towr_plus_trajectory.py --file=data/atlas_forward_walk.yaml --crbi_model_path=data/tf_model/atlas_crbi```
- Replay the optimized trajectory with the robot:<br/>
```$ python simulator/pybullet/atlas_kinematics_main.py --file=data/atlas_forward_walk.yaml```

## Implemented Features
### Planner
- Divergent Component of Motion: [paper](https://ieeexplore.ieee.org/document/7063218) | [code](https://github.com/junhyeokahn/PyPnC/tree/master/pnc/planner/locomotion/dcm_planner)
- TOWR+: paper | [code](https://github.com/junhyeokahn/PyPnC/tree/master/pnc/planner/locomotion/towr_plus)
### Controller
- Implicite Hierarchical Whole Body Controller: paper | [code](https://github.com/junhyeokahn/PyPnC/tree/master/pnc/wbc)
