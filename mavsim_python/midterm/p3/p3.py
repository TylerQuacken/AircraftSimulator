# problem 3

import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
# from scipy import linalg

### Part 1 ###

A_lon = np.zeros([5, 5])
A_lon[:3, :3] = np.array([[-0.00643, 0.0263, 0], [-0.0941, -0.624, 820],
                          [-0.000222, -0.00153, -0.668]])
A_lon[0, 3] = -32.2
A_lon[3, 2] = 1
A_lon[4, 1] = -1
A_lon[4, 3] = 830

B_lon = np.array([0, -32.7, -2.08, 0, 0])

eVal, eVec = np.linalg.eig(A_lon)
print("eigenvals: {}".format(eVal))

wn1 = np.sqrt(np.abs(eVal[1] * eVal[2])**2)
damp1 = np.abs(eVal[1] + eVal[2]) / wn1
wn2 = np.sqrt(np.abs(eVal[3] * eVal[4])**2)
damp2 = np.abs(eVal[3] + eVal[4]) / wn2

print("wn1: {}".format(wn1))
print("damp1: {}".format(damp1))
print("wn2: {}".format(wn2))
print("damp2: {}".format(damp2))

### Part 2 ###

# transfer function parameters
a_theta1 = 0.668
a_theta2 = 1.27
a_theta3 = -2.08
Va = 830.0

# control gains
damp_th = 0.707
wn_th = 10.0
kdth = (2 * damp_th * wn_th - a_theta1) / a_theta3
kpth = (wn_th**2 - a_theta2) / a_theta3

k_thDC = kpth * a_theta3 / (a_theta2 + kpth * a_theta3)

damp_h = 1.2
wn_h = wn_th / 20.0
kph = 2 * damp_h * wn_h / (k_thDC * Va)
kih = wn_h**2 / (k_thDC * Va)

# successive loop closure to form transfer function from h_c to h
# transfer function from del_e to q
q_del_e = ctrl.tf([a_theta3, 0], [1, a_theta1, a_theta2])

# close pitch rate loop to get del_e_c to q
q_del_e_c = ctrl.feedback(q_del_e, kdth)

# integrator 1/s to get theta from q
integrator = ctrl.tf(1, [1, 0])

# transfer function from theta_c to theta
th_th_c = ctrl.minreal(ctrl.feedback(kpth * q_del_e_c * integrator, 1))

# implementation of altitude PI control
pi_control = ctrl.tf([kph, kih], [1, 0])

# close PI feedback loop on altitude to get h_c to h
h_h_c = ctrl.feedback(pi_control * th_th_c * Va * integrator, 1)

# (note this h_h_c transfer function does not approximate th_th_c as KthDC
# instead it contains the full linearlized altitude dynamics)

# step response
t = 0.1 * np.arange(600)
t, h = ctrl.step_response(h_h_c, t)  # response to unit step command

altitude_step_size = 100.0

plt.figure(1)
# scale unit step response by the step size
plt.plot(t, altitude_step_size * h)
plt.xlabel('time (s)')
plt.ylabel('altitude (ft)')
plt.grid()
plt.axis([0, 60, 0, 120])
plt.title('Altitude step response for Boeing 747')
plt.show()
