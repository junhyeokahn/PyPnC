class Base(object):
    def __init__(self):
        self.a = 1
        

class Child(Base):
    def __init__(self):
        super(Child, self).__init__()
        print(self.a)
        

child = Child()
