"""
autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
"""
import sys
import numpy as np
import parameters.control_parameters as AP
from tools.transfer_function import transferFunction
from tools.wrap import wrap
from chap6.pi_control import PIControl
from chap6.pd_control_with_rate import PDControlWithRate
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
from chap5.model_coef import u_trim, Va_trim
sys.path.append('..')


class Autopilot:
    def __init__(self, ts_control):
        # instantiate lateral controllers
        self.roll_from_aileron = PDControlWithRate(kp=AP.roll_kp,
                                                   kd=AP.roll_kd,
                                                   limit=np.radians(45))
        self.course_from_roll = PIControl(kp=AP.course_kp,
                                          ki=AP.course_ki,
                                          Ts=ts_control,
                                          limit=np.radians(30))
        self.yaw_damper = transferFunction(
            num=np.array([[AP.yaw_damper_kr, 0]]),
            den=np.array([[1, AP.yaw_damper_p_wo]]),
            Ts=ts_control)

        # instantiate longitudinal controllers
        self.pitch_from_elevator = PDControlWithRate(kp=AP.pitch_kp,
                                                     kd=AP.pitch_kd,
                                                     limit=np.radians(45))
        self.altitude_from_pitch = PIControl(kp=AP.altitude_kp,
                                             ki=AP.altitude_ki,
                                             Ts=ts_control,
                                             limit=np.radians(30))
        self.airspeed_from_throttle = PIControl(kp=AP.airspeed_throttle_kp,
                                                ki=AP.airspeed_throttle_ki,
                                                Ts=ts_control,
                                                limit=1.0)
        self.commanded_state = MsgState()
        self.Va_trim = Va_trim

    def update(self, cmd, state):

        # lateral autopilot
        chi_c = wrap(cmd.course_command, state.chi)
        phi_c = cmd.phi_feedforward + self.course_from_roll.update(
            chi_c, state.chi)
        phi_c = self.saturate(phi_c, -np.radians(30), np.radians(30))
        delta_a = self.roll_from_aileron.update(phi_c, state.phi, state.p)
        delta_r = self.yaw_damper.update(state.r)

        # longitudinal autopilot
        # saturate the altitude command
        alt_band = AP.altitude_zone
        altitude_c = self.saturate(cmd.altitude_command,
                                   state.altitude - alt_band,
                                   state.altitude + alt_band)
        theta_c = self.altitude_from_pitch.update(altitude_c, state.altitude)
        delta_e = self.pitch_from_elevator.update(theta_c, state.theta,
                                                  state.q)
        delta_t = self.airspeed_from_throttle.update(
            cmd.airspeed_command - self.Va_trim, state.Va - self.Va_trim)
        delta_t = self.saturate(delta_t + u_trim.item(3), 0., 1.)

        # construct output and commanded states
        delta = MsgDelta(elevator=delta_e,
                         aileron=delta_a,
                         rudder=delta_r,
                         throttle=delta_t)
        self.commanded_state.altitude = cmd.altitude_command
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = phi_c
        self.commanded_state.theta = theta_c
        self.commanded_state.chi = cmd.course_command
        return delta, self.commanded_state

    def saturate(self, input, low_limit, up_limit):
        if input <= low_limit:
            output = low_limit
        elif input >= up_limit:
            output = up_limit
        else:
            output = input
        return output
