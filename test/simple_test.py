class Base(object):
    def __init__(self):
        self.a = 1
        

class Child(Base):
    def __init__(self):
        super(Child, self).__init__()
        print(self.a)
        

child = Child()

import numpy as np

if True:
    aa = np.array([1])
print(aa)
