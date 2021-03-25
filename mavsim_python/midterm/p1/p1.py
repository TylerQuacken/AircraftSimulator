import sys
import numpy as np
sys.path.append('../../')
from tools.rotations import Euler2Quaternion

u = 18
v = 2
w = -3
phi = np.radians(-30)
theta = np.radians(-15)
psi = np.radians(60)

quat = Euler2Quaternion(phi, theta, psi).squeeze()

print(quat)

R1 = np.array([[np.cos(psi), np.sin(psi), 0], [-np.sin(psi),
                                               np.cos(psi), 0], [0, 0, 1]])
R2 = np.array([[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0],
               [np.sin(theta), 0, np.cos(theta)]])

R3 = np.array([[1, 0, 0], [0, np.cos(phi), np.sin(phi)],
               [0, -np.sin(phi), np.cos(phi)]])

vb = np.array([u, v, w])

v2 = R3.T @ vb
v1 = R2.T @ v2
vw = R1.T @ v1

print(vw)
