import numpy as np

class boundary:
    def __init__(self):
        self.A_bounds = None
        self.b_bounds = None

    def setA(self, A):
        self.A_bounds = A

    def setB(self, B):
        self.b_bounds = B

    def getA(self):
        return self.A_bounds

    def getB(self):
        return self.b_bounds


def fromBounds(lb, ub):
    """
    Return a new Polyhedron representing an n-dimensional box spanning
    from [lb] to [ub]
    """
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    p = boundary()
    p.setA(np.vstack((-np.eye(lb.size), np.eye(lb.size))))
    p.setB(np.vstack((-lb, ub)))
    return p

# For backward compatibility
from_bounds = fromBounds

