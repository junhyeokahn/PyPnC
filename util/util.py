import numpy as np
import yaml


def pretty_print(ob):
    if type(ob) is dict:
        print(yaml.dump(ob, default_flow_style=False))
