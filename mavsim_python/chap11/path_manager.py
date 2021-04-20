import numpy as np
import sys
sys.path.append('..')
# from chap11.dubins_parameters import DubinsParameters
from message_types.msg_path import MsgPath


class PathManager:
    def __init__(self):
        # message sent to path follower
        self.path = MsgPath()
        # pointers to previous, current, and next waypoints
        self.ptr_previous = 0
        self.ptr_current = 1
        self.ptr_next = 2
        self.num_waypoints = 0
        self.halfspace_n = np.inf * np.ones((3, 1))
        self.halfspace_r = np.inf * np.ones((3, 1))
        # state of the manager state machine
        self.manager_state = 1
        self.manager_requests_waypoints = True
        # self.dubins_path = DubinsParameters()

    def update(self, waypoints, radius, state):
        if waypoints.num_waypoints == 0:
            self.manager_requests_waypoints = True
        if self.manager_requests_waypoints is True \
                and waypoints.flag_waypoints_changed is True:
            self.manager_requests_waypoints = False
        if waypoints.type == 'straight_line':
            self.line_manager(waypoints, state)
        elif waypoints.type == 'fillet':
            self.fillet_manager(waypoints, radius, state)
        elif waypoints.type == 'dubins':
            # self.dubins_manager(waypoints, radius, state)
            raise NotImplementedError
        else:
            print('Error in Path Manager: Undefined waypoint type.')
        return self.path

    def line_manager(self, waypoints, state):
        mav_pos = np.array([[state.north, state.east, -state.altitude]]).T
        # if the waypoints have changed, update the waypoint pointer
        if waypoints.flag_waypoints_changed is True:
            waypoints.flag_manager_requests_waypoints = False
            waypoints.flag_waypoints_changed = False
            self.num_waypoints = waypoints.num_waypoints
            self.initialize_pointers()
            self.construct_line(waypoints)

        # state machine for line path
        # if its in the half space
        if self.inHalfSpace(mav_pos):
            self.increment_pointers()
            self.construct_line(waypoints)
            if self.ptr_current == 0:
                self.manager_requests_waypoints = True

    def fillet_manager(self, waypoints, radius, state):
        mav_pos = np.array([[state.north, state.east, -state.altitude]]).T
        # if the waypoints have changed, update the waypoint pointer
        if waypoints.flag_waypoints_changed is True:
            # waypoints.flag_manager_requests_waypoints = False
            waypoints.flag_waypoints_changed = False
            self.num_waypoints = waypoints.num_waypoints
            self.initialize_pointers()
            self.construct_fillet_line(waypoints, radius)
            self.manager_state = 1

        # state machine for fillet path
        if self.manager_state == 1:
            # follow straight line path from previous to current
            if self.inHalfSpace(mav_pos):
                # entered into H1
                self.construct_fillet_circle(waypoints, radius)
                self.manager_state = 3
        elif self.manager_state == 2:
            # follow orbit until out of H2
            if not self.inHalfSpace(mav_pos):
                self.manager_state = 3
        elif self.manager_state == 3:
            # follow orbit from previous->current to current->next
            if self.inHalfSpace(mav_pos):
                # entered into half plane H2
                self.increment_pointers()
                self.construct_fillet_line(waypoints, radius)
                self.manager_state = 1
                # requests new waypoints
                if self.ptr_current == 0:
                    self.manager_requests_waypoints = True

    def initialize_pointers(self):
        if self.num_waypoints >= 3:
            self.ptr_previous = 0
            self.ptr_current = 1
            self.ptr_next = 2
        else:
            print('Error Path Manager: need at least three waypoints')

    def increment_pointers(self):
        self.ptr_previous = self.ptr_current
        self.ptr_current = self.ptr_next
        # make next or current 9999 if invalid
        self.ptr_next = (self.ptr_next + 1) % self.num_waypoints
        print("Prev: {}\tCurrent: {}\tNext: {}".format(self.ptr_previous,
                                                       self.ptr_current,
                                                       self.ptr_next))
        # if self.ptr_next > self.num_waypoints - 1:
        #     self.ptr_next = 9999
        # if self.ptr_current > self.num_waypoints - 1:
        #     self.ptr_current = 9999

    def inHalfSpace(self, pos):
        if (pos - self.halfspace_r).T @ self.halfspace_n >= 0:
            return True
        else:
            return False

    def construct_line(self, waypoints):
        previous = waypoints.ned[:, self.ptr_previous:self.ptr_previous + 1]
        if self.ptr_current == 9999:
            # use path.line_direction to fly straight
            current = previous + 100 * self.path.line_direction
        else:
            # get current waypoint from waypoints.ned
            current = waypoints.ned[:, self.ptr_current].reshape(-1, 1)
        if self.ptr_next == 9999:
            # go straight for even longer
            next = previous + 200 * self.path.line_direction
        else:
            next = waypoints.ned[:, self.ptr_next].reshape(-1, 1)
        # update path variables
        self.path.type = 'line'
        self.path.airspeed = waypoints.airspeed.item(self.ptr_current)
        self.path.line_origin = previous
        qPrev = current - previous
        qPrev /= np.linalg.norm(qPrev)
        self.path.line_direction = qPrev
        qNext = next - current
        qNext /= np.linalg.norm(qNext)

        self.halfspace_n = qPrev + qNext
        self.halfspace_n /= np.linalg.norm(self.halfspace_n)
        self.halfspace_r = current
        self.path.plot_updated = False

    def construct_fillet_line(self, waypoints, radius):
        previous = waypoints.ned[:, self.ptr_previous:self.ptr_previous + 1]
        if self.ptr_current == 9999:
            current = previous + 100 * self.path.line_direction
        else:
            current = waypoints.ned[:, self.ptr_current].reshape(-1, 1)
        if self.ptr_next == 9999:
            next = previous + 200 * self.path.line_direction
        else:
            next = waypoints.ned[:, self.ptr_next].reshape(-1, 1)
        # update path variables same
        self.path.type = 'line'
        self.path.airspeed = waypoints.airspeed.item(self.ptr_current)
        self.path.line_origin = previous
        qPrev = current - previous
        qPrev /= np.linalg.norm(qPrev)
        self.path.line_direction = qPrev
        qNext = next - current
        qNext /= np.linalg.norm(qNext)
        varphi = np.arccos(-qPrev.T @ qNext)

        self.halfspace_n = qPrev
        self.halfspace_r = current - (radius / np.tan(varphi / 2)) * qPrev
        self.path.plot_updated = False

        # halfspace_n is just previous
        # halfspace_r should be current minus a bit

    def construct_fillet_circle(self, waypoints, radius):
        previous = waypoints.ned[:, self.ptr_previous:self.ptr_previous + 1]
        if self.ptr_current == 9999:
            current = previous + 100 * self.path.line_direction
        else:
            current = waypoints.ned[:, self.ptr_current].reshape(-1, 1)
        if self.ptr_next == 9999:
            next = previous + 200 * self.path.line_direction
        else:
            next = waypoints.ned[:, self.ptr_next].reshape(-1, 1)
        #update path variables
        self.path.type = 'orbit'
        self.path.airspeed = waypoints.airspeed.item(self.ptr_current)
        qPrev = current - previous
        qPrev /= np.linalg.norm(qPrev)
        qNext = next - current
        qNext /= np.linalg.norm(qNext)
        varphi = np.arccos(-qPrev.T @ qNext)
        # set orbit_center
        qC = qPrev - qNext
        qC /= np.linalg.norm(qC)
        self.path.orbit_center = current - (radius /
                                            (np.sin(varphi / 2) + 0.001)) * qC
        self.path.orbit_radius = radius
        # set orbit direction as CW (lam=1) or CCW (lam=2)
        if np.sign(
                qPrev.item(0) * qNext.item(1) -
                qPrev.item(1) * qNext.item(0)) > 0:
            self.path.orbit_direction = 'CW'
        else:
            self.path.orbit_direction = 'CCW'

        self.halfspace_n = qNext
        self.halfspace_r = current + (radius /
                                      (np.tan(varphi / 2) + 0.001)) * qNext
        self.path.plot_updated = False

    # def dubins_manager(self, waypoints, radius, state):
    #     mav_pos = np.array([[state.north, state.east, -state.altitude]]).T
    #     # if the waypoints have changed, update the waypoint pointer

    #     # state machine for dubins path

    # def construct_dubins_circle_start(self, waypoints, dubins_path):
    #     #update path variables
    #     pass

    # def construct_dubins_line(self, waypoints, dubins_path):
    #     #update path variables
    #     pass

    # def construct_dubins_circle_end(self, waypoints, dubins_path):
    #     #update path variables
    #     pass
