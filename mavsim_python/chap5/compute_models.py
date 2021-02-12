"""
compute_ss_model
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/4/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
from scipy.optimize import minimize
from tools.rotations import Euler2Quaternion, Quaternion2Euler
import parameters.aerosonde_parameters as MAV
from parameters.simulation_parameters import ts_simulation as Ts
from message_types.msg_delta import MsgDelta


def compute_model(mav, trim_state, trim_input):
    A_lon, B_lon, A_lat, B_lat = compute_ss_model(mav, trim_state, trim_input)
    breakpoint()
    Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, \
        a_V1, a_V2, a_V3 = compute_tf_model(mav, trim_state, trim_input)

    # write transfer function gains to file
    file = open('model_coef.py', 'w')
    file.write('import numpy as np\n')
    file.write(
        'x_trim = np.array([[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]]).T\n'
        % (trim_state.item(0), trim_state.item(1), trim_state.item(2),
           trim_state.item(3), trim_state.item(4), trim_state.item(5),
           trim_state.item(6), trim_state.item(7), trim_state.item(8),
           trim_state.item(9), trim_state.item(10), trim_state.item(11),
           trim_state.item(12)))
    file.write('u_trim = np.array([[%f, %f, %f, %f]]).T\n' %
               (trim_input.elevator, trim_input.aileron, trim_input.rudder,
                trim_input.throttle))
    file.write('Va_trim = %f\n' % Va_trim)
    file.write('alpha_trim = %f\n' % alpha_trim)
    file.write('theta_trim = %f\n' % theta_trim)
    file.write('a_phi1 = %f\n' % a_phi1)
    file.write('a_phi2 = %f\n' % a_phi2)
    file.write('a_theta1 = %f\n' % a_theta1)
    file.write('a_theta2 = %f\n' % a_theta2)
    file.write('a_theta3 = %f\n' % a_theta3)
    file.write('a_V1 = %f\n' % a_V1)
    file.write('a_V2 = %f\n' % a_V2)
    file.write('a_V3 = %f\n' % a_V3)
    file.write(
        'A_lon = np.array([\n    [%f, %f, %f, %f, %f],\n    '
        '[%f, %f, %f, %f, %f],\n    '
        '[%f, %f, %f, %f, %f],\n    '
        '[%f, %f, %f, %f, %f],\n    '
        '[%f, %f, %f, %f, %f]])\n' %
        (A_lon[0][0], A_lon[0][1], A_lon[0][2], A_lon[0][3], A_lon[0][4],
         A_lon[1][0], A_lon[1][1], A_lon[1][2], A_lon[1][3], A_lon[1][4],
         A_lon[2][0], A_lon[2][1], A_lon[2][2], A_lon[2][3], A_lon[2][4],
         A_lon[3][0], A_lon[3][1], A_lon[3][2], A_lon[3][3], A_lon[3][4],
         A_lon[4][0], A_lon[4][1], A_lon[4][2], A_lon[4][3], A_lon[4][4]))
    file.write('B_lon = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' % (
                   B_lon[0][0],
                   B_lon[0][1],
                   B_lon[1][0],
                   B_lon[1][1],
                   B_lon[2][0],
                   B_lon[2][1],
                   B_lon[3][0],
                   B_lon[3][1],
                   B_lon[4][0],
                   B_lon[4][1],
               ))
    file.write(
        'A_lat = np.array([\n    [%f, %f, %f, %f, %f],\n    '
        '[%f, %f, %f, %f, %f],\n    '
        '[%f, %f, %f, %f, %f],\n    '
        '[%f, %f, %f, %f, %f],\n    '
        '[%f, %f, %f, %f, %f]])\n' %
        (A_lat[0][0], A_lat[0][1], A_lat[0][2], A_lat[0][3], A_lat[0][4],
         A_lat[1][0], A_lat[1][1], A_lat[1][2], A_lat[1][3], A_lat[1][4],
         A_lat[2][0], A_lat[2][1], A_lat[2][2], A_lat[2][3], A_lat[2][4],
         A_lat[3][0], A_lat[3][1], A_lat[3][2], A_lat[3][3], A_lat[3][4],
         A_lat[4][0], A_lat[4][1], A_lat[4][2], A_lat[4][3], A_lat[4][4]))
    file.write('B_lat = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' % (
                   B_lat[0][0],
                   B_lat[0][1],
                   B_lat[1][0],
                   B_lat[1][1],
                   B_lat[2][0],
                   B_lat[2][1],
                   B_lat[3][0],
                   B_lat[3][1],
                   B_lat[4][0],
                   B_lat[4][1],
               ))
    file.write('Ts = %f\n' % Ts)
    file.close()


def compute_tf_model(mav, trim_state, trim_input):
    # trim values
    mav._state = trim_state
    mav._update_velocity_data()
    Va_trim = mav._Va
    alpha_trim = mav._alpha
    phi, theta_trim, psi = Quaternion2Euler(trim_state[6:10])

    # define transfer function constants
    a_phi1 = 0.5*MAV.rho*Va_trim**2 * MAV.S_wing * MAV.b **2 * MAV.C_p_p / (2*Va_trim)
    a_phi2 = 0.5*MAV.rho*Va_trim**2 * MAV.S_wing * MAV.b * MAV.C_p_delta_a
    const = 0.5*MAV.rho*Va_trim**2*MAV.c*MAV.S_wing/MAV.Jy
    a_theta1 = -const * MAV.C_m_q *MAV.c /(2.*Va_trim)
    a_theta2 = -const * MAV.C_m_alpha
    a_theta3 = const * MAV.C_m_delta_e

    # Compute transfer function coefficients using new propulsion model
    a_V1 = MAV.rho * Va_trim * MAV.S_prop / MAV.mass * (MAV.C_D_0 + MAV.C_D_alpha*alpha_trim + MAV.C_D_delta_e * trim_input.elevator) - dT_dVa(mav, trim_input.throttle, Va_trim) / MAV.mass
    a_V2 = dT_ddelta_t(mav, trim_input.throttle, Va_trim) / MAV.mass
    a_V3 = MAV.gravity * np.cos(theta_trim - alpha_trim)

    return Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, a_V1, a_V2, a_V3

    
def compute_ss_model(mav, trim_state, trim_input):
    x_euler = euler_state(trim_state)
    A = df_dx(mav, x_euler, trim_input)
    B = df_du(mav, x_euler, trim_input)
    # extract longitudinal states (u, w, q, theta, pd) and change pd to h
    lon_mask = np.array([3, 5, 10, 7, 2])
    lon_mask2 = np.array([0, 3])
    A_lon = A[lon_mask, :][:, lon_mask]
    B_lon = B[lon_mask, :][:, lon_mask2]
    # extract lateral states (v, p, r, phi, psi)
    lat_mask = np.array([4, 9, 11, 6, 8])
    lat_mask2 = np.array([1, 2])
    A_lat = A[lat_mask, :][:, lat_mask]
    B_lat = B[lat_mask, :][:, lat_mask2]
    return A_lon, B_lon, A_lat, B_lat


def euler_state(x_quat):
    # convert state x with attitude represented by quaternion
    # to x_euler with attitude represented by Euler angles
    e = x_quat[6:10]
    phi, theta, psi = Quaternion2Euler(e)
    x_euler = np.zeros([12, 1])
    x_euler[:6, :] = x_quat[:6, :]
    x_euler[6, :] = phi
    x_euler[7, :] = theta
    x_euler[8, :] = psi
    x_euler[9:, :] = x_quat[10:, :]
    return x_euler


def quaternion_state(x_euler):
    # convert state x_euler with attitude represented by Euler angles
    # to x_quat with attitude represented by quaternions
    quat = Euler2Quaternion(x_euler[6], x_euler[7], x_euler[8])
    x_quat = np.zeros([13, 1])
    x_quat[:6, :] = x_euler[:6, :]
    x_quat[6:10, 0] = quat.flatten()
    x_quat[10:, :] = x_euler[9:, :]
    return x_quat


def f_euler(mav, x_euler, delta):
    # return 12x1 dynamics (as if state were Euler state)
    # compute f at euler_state
    x_quat = quaternion_state(x_euler)
    mav._state = x_quat.reshape([13, 1])
    mav._update_velocity_data()
    forces_moments = mav._forces_moments(delta)
    xDot = mav._derivatives(x_quat, forces_moments)
    de_dt = xDot[6:10, :]
    quat = x_quat[6:10, :]
    dE_de = np.zeros([3, 4])

    perturb = 0.001

    for i in range(4):
        de = np.zeros([4, 1])
        de[i, :] = perturb
        quat_p = quat + de
        quat_p /= np.sum(quat_p)
        E_p = np.array(Quaternion2Euler(quat_p))
        E = np.array(Quaternion2Euler(quat))
        dE_de[:, i] = (E_p - E) / perturb

    dE_dt = dE_de @ de_dt

    f_euler_ = np.zeros([12, 1])
    f_euler_[:6, :] = xDot[:6, :]
    f_euler_[6:9, :] = dE_dt
    f_euler_[9:, :] = xDot[10:, :]

    return f_euler_


def df_dx(mav, x_euler, delta):
    # take partial of f_euler with respect to x_euler
    A = np.zeros([12, 12])
    perturb = 0.001
    for i in range(12):
        dx = np.zeros([12, 1])
        dx[i, :] = perturb
        f = f_euler(mav, x_euler, delta)
        f_p = f_euler(mav, x_euler + dx, delta)
        dfx = (f_p - f) / perturb
        A[:, i] = dfx.flatten()
        
    return A


def df_du(mav, x_euler, delta):
    # take partial of f_euler with respect to input

    B = np.zeros([12, 4])
    perturb = 0.001
    for i in range(4):
        ddelta = np.zeros([4, 1])
        ddelta[i, :] = perturb
        delta_p_array = delta.to_array() + ddelta
        delta_p = MsgDelta()
        delta_p.from_array(delta_p_array)
        f = f_euler(mav, x_euler, delta)
        f_p = f_euler(mav, x_euler, delta_p)
        dfx = (f_p - f) / perturb
        B[:, i] = dfx.flatten()

    return B


def dT_dVa(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to Va
    eps = 0.001
    T_eps, Q_eps = mav._motor_thrust_torque(Va + eps, delta_t)
    T, Q = mav._motor_thrust_torque(Va, delta_t)
    return (T_eps - T) / eps

def dT_ddelta_t(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to delta_t
    eps = 0.001
    T_eps, Q_eps = mav._motor_thrust_torque(Va, delta_t + eps)
    T, Q = mav._motor_thrust_torque(Va, delta_t)
    return (T_eps - T) / eps
