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
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
from chap5.model_coef import u_trim, Va_trim
from chap6.pi_control import PIControl
from chap6.pd_control_with_rate import PDControlWithRate
from psompc.PSOMPC import PSOMPC
from rad_models.UAV import UAV
from IPython import embed
from ParticleVisualizer import ParticleVisualizer
sys.path.append('..')


class Autopilot:
    def __init__(self, ts_control):
        """
        
        """
        self.dt = ts_control
        self.learnedSys = UAV()
        self.numStates = self.learnedSys.numStates
        self.numInputs = self.learnedSys.numInputs
        # phi, theta, psi form. Va is appended as the last element
        self.Q = 1.0 * np.diag(
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.Qf = self.Q
        self.R = 100.0 * np.diag([1.0, 0.0, 0.0, 1.0])
        experimentalFlags = dict(zeroVelAtBoundary=True,
                                 numSwarms=1,
                                 chargedParticles=0.5)

        self.controller = PSOMPC(self.learnedSys.forward_simulate_dt,
                                 self.evaluate_cost,
                                 self.numStates,
                                 self.numInputs,
                                 socialWeight=0.01,
                                 personalWeight=1.0,
                                 uMin=self.learnedSys.uMin,
                                 uMax=self.learnedSys.uMax,
                                 horizon=30,
                                 numSims=500,
                                 numKnotPoints=4,
                                 dt=self.dt,
                                 useGPU=False,
                                 experimentalFlags=experimentalFlags)

        # self.particleVisualizer = ParticleVisualizer(self.controller)

        # xVa is x with euler angles and Va appended as the last element
        self.xVaGoal = np.array(
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.uGoal = np.zeros([4])
        self.trimU = u_trim.flatten()
        self.prevU = u_trim.flatten()
        self.inertiaMax = 1.5
        self.inertiaMin = 0.4
        self.socialMax = 0.02
        self.personalMax = 1.0
        self.errMax = None

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

    def evaluate_cost(self, x, u, xVaGoal, uGoal, finalTimestep=False):
        Va = np.sqrt(x[:, 3]**2 + x[:, 4]**2 + x[:, 5]**2)
        Va = np.expand_dims(Va, 1)
        # append Va as dim 12 of x
        x = np.concatenate([x, Va], axis=1)

        # Make X, U, batches of column vectors
        X_ = np.expand_dims(x - xVaGoal, -1)
        U_ = np.expand_dims(u - uGoal, -1)
        if finalTimestep:
            Qx = np.abs(self.Qf @ X_**2)
            cost = np.sum(Qx, axis=1)
        else:
            Qx = np.abs(self.Q @ X_**2)
            Ru = np.abs(self.R @ U_**2)
            cost = np.sum(Qx, axis=1) + np.sum(Ru, axis=1)
        return cost.squeeze()

    def update(self, cmd, state, mavstate):
        """
        mavstate is the self._state variable from the mav dynamics. We need to pick off u, v, w from here, since no other part of the simulator keeps track of that
        """
        # cmd has four vars: airspeed_command, course_command, altitude_command, phi_feedforward
        # ignore phi_feedforward for now

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

        # get x from state
        x = np.array([
            state.north, state.east, -state.altitude,
            mavstate.item(3),
            mavstate.item(4),
            mavstate.item(5), state.phi, state.theta, state.chi, state.p,
            state.q, state.r
        ])

        # set x_goal from cmd
        chi_c = wrap(cmd.course_command, state.chi)
        alt_band = AP.altitude_zone
        altitude_c = self.saturate(cmd.altitude_command,
                                   state.altitude - alt_band,
                                   state.altitude + alt_band)
        altitude_c = cmd.altitude_command
        self.xVaGoal[2] = altitude_c
        self.xVaGoal[8] = chi_c
        self.xVaGoal[12] = cmd.airspeed_command

        # get params for MPC
        errStates = [2]
        xVa = np.append(x, state.Va)
        err = np.linalg.norm(xVa[errStates] - self.xVaGoal[errStates])
        if self.errMax is None:
            self.errMax = err
        inertia = err / self.errMax * (self.inertiaMax -
                                       self.inertiaMin) + self.inertiaMin
        self.controller.personalWeight = err * self.personalMax / self.errMax
        self.controller.socialWeight = (1 - err / self.errMax) * self.socialMax

        self.learnedSys.uFedIn = np.array([0.0, delta_a, delta_r, 0.0])

        for i in range(1):
            u = self.controller.solve_for_next_u(x,
                                                 self.xVaGoal,
                                                 self.prevU,
                                                 self.trimU,
                                                 inertia=inertia)
        # u = self.prevU
        print("u: \t", u[[0, 3]])
        self.prevU = u

        # construct output and commanded states
        delta = MsgDelta(elevator=u[0],
                         aileron=delta_a,
                         rudder=delta_r,
                         throttle=u[3])
        self.commanded_state.altitude = cmd.altitude_command
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = phi_c
        self.commanded_state.theta = theta_c * 0
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
