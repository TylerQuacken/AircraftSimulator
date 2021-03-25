import numpy as np
from math import sin, cos
import sys

sys.path.append('..')
from message_types.msg_autopilot import MsgAutopilot
from tools.wrap import wrap


class PathFollower:
    def __init__(self):
        self.chi_inf = np.radians(
            60)  # approach angle for large distance from straight-line path
        self.k_path = 0.05  # proportional gain for straight-line path following
        self.k_orbit = 10.0  # proportional gain for orbit following
        self.gravity = 9.8
        self.autopilot_commands = MsgAutopilot()  # message sent to autopilot

    def update(self, path, state):
        if path.type == 'line':
            self._follow_straight_line(path, state)
        elif path.type == 'orbit':
            self._follow_orbit(path, state)
        return self.autopilot_commands

    def _follow_straight_line(self, path, state):
        self.autopilot_commands.airspeed_command = path.airspeed
        r = path.line_origin
        p = np.array([[state.north, state.east, -state.altitude]]).T
        # chi_q = np.arctan2(path.line_direction.item(1),
        #                    path.line_direction.item(0))
        chi_q = np.arccos(path.line_direction.item(0))

        while chi_q - state.chi < -np.pi:
            chi_q += 2 * np.pi
        while chi_q - state.chi > np.pi:
            chi_q -= 2 * np.pi

        R = np.array([[np.cos(chi_q), np.sin(chi_q), 0],
                      [-np.sin(chi_q), np.cos(chi_q), 0], [0, 0, 1]])
        e_p = R @ (p - r)
        # course command
        self.autopilot_commands.course_command = chi_q - self.chi_inf * (
            2 / np.pi) * np.arctan(self.k_path * e_p.item(1))
        # altitude command
        k = np.array([[0, 0, 1]]).T
        q = path.line_direction
        n = np.cross(k.squeeze(), q.squeeze()).reshape(-1, 1)
        n /= np.linalg.norm(n)
        s = e_p - np.dot(e_p.squeeze(), n.squeeze()) * n

        self.autopilot_commands.altitude_command = -r.item(2) - np.sqrt(
            s.item(0)**2 +
            s.item(1)**2) * (q.item(2) / np.sqrt(q.item(0)**2 + q.item(1)**2))
        # feedforward roll angle for straight line is zero
        self.autopilot_commands.phi_feedforward = 0

    def _follow_orbit(self, path, state):
        if path.orbit_direction == 'CW':
            direction = 1.0
        else:
            direction = -1.0
        # airspeed command
        self.autopilot_commands.airspeed_command = path.airspeed
        # distance from orbit center
        pn = state.north
        pe = state.east
        cn = path.orbit_center.item(0)
        ce = path.orbit_center.item(1)
        d_vec = np.array([state.north, state.east, -state.altitude
                          ]) - path.orbit_center.flatten()
        d = np.linalg.norm(d_vec[:2])
        # compute wrapped version of angular position on orbit
        varphi = np.arctan2(pe - ce, pn - cn)
        while varphi - state.chi < -np.pi:
            varphi += 2 * np.pi
        while varphi - state.chi > np.pi:
            varphi -= 2 * np.pi

        chi_0 = varphi + direction * np.pi / 2
        # course command
        self.autopilot_commands.course_command = chi_0 + direction * np.arctan(
            self.k_orbit * (d - path.orbit_radius) / path.orbit_radius)
        # self.autopilot_commands.course_command = chi_0 + direction * (
        #     np.pi / 2 + np.arctan2(self.k_orbit * (d - path.orbit_radius),
        #                            self.k_orbit * path.orbit_radius))
        # altitude command
        self.autopilot_commands.altitude_command = -path.orbit_center.item(2)

        orbit_error = d - path.orbit_radius
        # roll feedforward command
        if orbit_error < 10:
            self.autopilot_commands.phi_feedforward = direction * np.arctan2(
                state.Va**2, (self.gravity * path.orbit_radius))
        else:
            self.autopilot_commands.phi_feedforward = 0
