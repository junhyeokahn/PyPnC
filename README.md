# PyPnC
Python Implementation of Planning and Control

## Requirements
- Install [dartpy](http://dartsim.github.io/install_dartpy_on_ubuntu.html)
- Install [ifopt](https://github.com/ethz-adrl/ifopt)
- Install the other dependencies
```
pip install -r requirements.txt.
```

## Running Experiments
### Whole Body Control Example
- Running WBC example
```
python simulator/pybullet/atlas_main.py
```

### Towr+ Example
- Make directories for data saving
```
mkdir data & mkdir video
```
- First compile Towr+ code
```
mkdir build && cd build && cmake ..
make -j4
```
- Then run an example
```
./atlas_two_step_yaml_test
```
- You can plot the result
```
python plot/plot_towr_plus_trajectory.py --file=data/atlas_two_step_yaml_test.yaml
```
