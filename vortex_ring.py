import numpy as np
from numpy.linalg import norm

def VRTXLINE(p, pt1, pt2, Gamma):
    p = np.array(p)
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)

    R1 = p-pt1
    R2 = p-pt2
    R0 = pt2-pt1
    R1xR2 = np.cross(R1,R2)
    R1xR2_norm = norm(R1xR2)

    r1 = norm(R1)
    r2 = norm(R2)

    K = Gamma/(4*np.pi*R1xR2_norm**2)*(np.dot(R0,R1)/r1 - np.dot(R0,R2)/r2)

    V = K*R1xR2

    u = V[0]
    v = V[1]
    w = V[2]

    return u, v, w

def VRTXRING(collocation, pt1, pt2, pt3, pt4, Gamma):

    u1, v1, w1 = VRTXLINE(collocation, pt1, pt2, Gamma)
    u2, v2, w2 = VRTXLINE(collocation, pt2, pt3, Gamma)
    u3, v3, w3 = VRTXLINE(collocation, pt3, pt4, Gamma)
    u4, v4, w4 = VRTXLINE(collocation, pt4, pt1, Gamma)

    u = u1 + u2 + u3 + u4
    v = v1 + v2 + v3 + v4
    w = w1 + w2 + w3 + w4

    return u, v, w

