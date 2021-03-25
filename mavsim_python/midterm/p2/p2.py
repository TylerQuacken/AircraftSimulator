import sys
import numpy as np
sys.path.append('../../')
from tools.rotations import Euler2Quaternion

u = 45
v = -2
w = 6
phi = np.radians(5)
theta = np.radians(10)
psi = np.radians(-27)
Vw = np.array([5, -1, 0])

quat = Euler2Quaternion(phi, theta, psi).squeeze()

R1 = np.array([[np.cos(psi), np.sin(psi), 0], [-np.sin(psi),
                                               np.cos(psi), 0], [0, 0, 1]])
R2 = np.array([[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0],
               [np.sin(theta), 0, np.cos(theta)]])

R3 = np.array([[1, 0, 0], [0, np.cos(phi), np.sin(phi)],
               [0, -np.sin(phi), np.cos(phi)]])

vb = np.array([u, v, w])

v2 = R3.T @ vb
v1 = R2.T @ v2
Vg = R1.T @ v1

Vab = Vg - Vw
Va = np.linalg.norm(Vab)

(ur, vr, wr) = tuple(Vab)
alpha = np.arctan2(wr, ur)
beta = np.arcsin(vr / Va)

gammaA = theta - alpha

print("alpha: {}".format(np.rad2deg(alpha)))
print("beta: {}".format(np.rad2deg(beta)))
print("Va: {}".format(Va))
print("gammaA: {}".format(np.rad2deg(gammaA)))
