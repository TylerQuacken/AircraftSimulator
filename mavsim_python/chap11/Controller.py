import numpy as np
from psompc.PSOMPC import PSOMPC
from rad_models.UAV import UAV
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from IPython import embed
from tqdm import tqdm
import sys
sys.path.append('..')
from message_types.msg_delta import MsgDelta
from message_types.msg_state import MsgState
from message_types.msg_autopilot import MsgAutopilot


class Controller():
    def __init__(self):
        sys = UAV()
        learnedSys = UAV()

        numStates = sys.numStates
        numInputs = sys.numInputs
        self.Q = 1.0 * np.diag(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.Qf = self.Q
        self.R = 0.0000 * np.diag([1.0, 1.0, 1.0, 1.0])
        self.uPrev = np.zeros([4])

        experimentalFlags = dict(zeroVelAtBoundary=True,
                                 numSwarms=2,
                                 chargedParticles=0.5)

        self.controller = PSOMPC(learnedSys.forward_simulate_dt,
                                 self.evaluate_cost,
                                 numStates,
                                 numInputs,
                                 socialWeight=0.01,
                                 personalWeight=1.0,
                                 horizon=15,
                                 uMin=sys.uMin,
                                 uMax=sys.uMax,
                                 useGPU=True,
                                 numSims=300,
                                 numKnotPoints=3,
                                 topology="nSwarm",
                                 experimentalFlags=experimentalFlags)

    def evaluate_cost(self, x, u, xGoal, uGoal, finalTimestep=False):
        # Make X, U, batches of column vectors
        X_ = np.expand_dims(x - xGoal, -1)
        U_ = np.expand_dims(u - uGoal, -1)
        if finalTimestep:
            Qx = np.abs(self.Qf @ X_**2)
            cost = np.sum(Qx, axis=1)
        else:
            Qx = np.abs(self.Q @ X_**2)
            Ru = np.abs(self.R @ U_**2)
            cost = np.sum(Qx, axis=1) + np.sum(Ru, axis=1)
        return cost.squeeze()

    def update(self, commands, state):

        xGoal = np.zeros([13])
        # commands gives airspeed, course, altitude, phi_feedforward
        x = np.array([state.north,
                      state.east,
                      -state.altitude,
                      e0,
                      e1,
                      e2,
                      e3,
                      

        u = self.controller.solve_for_next_u(x,
                                             xGoal,
                                             self.uPrev,
                                             uGoal,
                                             inertia=inertia)
        delta = MsgDelta(u[0], u[1], u[2], u[3])
        commanded_state = MsgState()
        return delta, commanded_state
