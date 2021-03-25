import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

data = np.loadtxt("ufo_data.txt", delimiter=',')
tMeas_hist = data[:, 0]
F_hist = data[:, 1]
zMeas_hist = data[:, 2]

m = 10
b0 = 0.5
b1 = 1
g = 9.81
Ts = 0.05
N = 1
Tp = Ts / N
Q = np.diag([0.25**2, 0.25**2])
R = np.diag([0.1**2, 0])
P = np.eye(2)


def f(x, F):
    v = x[1]
    vDot = -b0 * v / m - b1 * abs(v) * v - g + F / m

    f = np.array([v, vDot])
    return f


def df_dx(xHat, u):
    vHat = xHat[1]
    df_dx = np.zeros([2, 2])
    df_dx[0, 1] = 1
    df_dx[1, 1] = -b0 / m - 2 * b1 * abs(vHat) / m
    return df_dx


def h(xHat, u):
    h = np.array([xHat[0], 0])
    return h


def dh_dx(xHat, u):
    dh_dx = np.zeros([2, 2])
    dh_dx[0, 0] = 1
    return dh_dx


x = np.array([20, 0.0])
tf = tMeas_hist[-1]
x_hist = np.zeros([int(tf / Tp), 2])
t_hist = np.zeros([int(tf / Tp)])

t = 0
lastMeas = -100
i = 0

while t < tf - 0.0001:
    index = np.argmax(tMeas_hist >= t - 0.001)
    F = F_hist[index]

    # prediction step
    x = x + Tp * f(x, F)
    A = df_dx(x, F)
    Ad = np.eye(2) + A * Tp + Tp**2 * A @ A / 2
    P = Ad @ P @ Ad.T + Tp**2 * Q

    if (t - lastMeas) >= Ts - 0.001:
        # Meas Update
        y = zMeas_hist[index]
        C = dh_dx(x, F)
        S_inv = np.linalg.pinv(R + C @ P @ C.T)
        L = P @ C.T @ S_inv
        P = ((np.eye(2) - L @ C) @ P @ (np.eye(2) - L @ C).T) + L @ R @ L.T
        x = x + L @ (y - h(x, F))

        lastMeas = t

    x_hist[i, :] = x
    t_hist[i] = t

    i += 1
    t += Tp

plt.figure(1)
plt.plot(tMeas_hist[::5], zMeas_hist[::5], 'y.')
plt.plot(t_hist, x_hist[:, 0])
plt.xlabel('Time (s)')
plt.ylabel('Z (m)')
plt.legend(['Measured z', 'Predicted z'])
plt.title("Predicted Z, Ts={}".format(Ts))
plt.grid()

plt.figure(2)
plt.plot(t_hist, x_hist[:, 1])
plt.xlabel('Time (s)')
plt.ylabel('v (m/2)')
plt.legend(['Predicted v'])
plt.title("Predicted v, Ts={}".format(Ts))
plt.grid()
plt.show()
