"""
mavsim_python: drawing tools
    - Beard & McLain, PUP, 2012
    - Update history:
        1/13/2021 - TWM
"""
import numpy as np
import pyqtgraph.opengl as gl
from tools.rotations import Euler2Rotation


class drawMav:
    def __init__(self, state, window):
        """
        Draw the Mav.

        The input to this function is a (message) class with properties that define the state.
        The following properties are assumed:
            state.north  # north position
            state.east  # east position
            state.altitude   # altitude
            state.phi  # roll angle
            state.theta  # pitch angle
            state.psi  # yaw angle
        """
        # get points that define the non-rotated, non-translated spacecraft and the mesh colors
        self.mav_points, self.mav_meshColors = self.get_points()

        mav_position = np.array([[state.north], [state.east],
                                 [-state.altitude]])  # NED coordinates
        # attitude of spacecraft as a rotation matrix R from body to inertial
        R = Euler2Rotation(state.phi, state.theta, state.psi)
        # rotate and translate points defining spacecraft
        rotated_points = self.rotate_points(self.mav_points, R)
        translated_points = self.translate_points(rotated_points, mav_position)

        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = self.points_to_mesh(translated_points)
        self.mav_body = gl.GLMeshItem(
            vertexes=mesh,  # defines the triangular mesh (Nx3x3)
            vertexColors=self.mav_meshColors,  # defines mesh colors (Nx1)
            drawEdges=True,  # draw edges between mesh elements
            smooth=False,  # speeds up rendering
            computeNormals=False)  # speeds up rendering
        window.addItem(self.mav_body)  # add body to plot

    def update(self, state):
        mav_position = np.array([[state.north], [state.east],
                                 [-state.altitude]])  # NED coordinates
        # attitude of spacecraft as a rotation matrix R from body to inertial
        R = Euler2Rotation(state.phi, state.theta, state.psi)
        # rotate and translate points defining spacecraft
        rotated_points = self.rotate_points(self.mav_points, R)
        translated_points = self.translate_points(rotated_points, mav_position)
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

        translated_points = R @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = self.points_to_mesh(translated_points)
        # draw spacecraft by resetting mesh using rotated and translated points
        self.mav_body.setMeshData(vertexes=mesh,
                                  vertexColors=self.mav_meshColors)

    def rotate_points(self, points, R):
        "Rotate points by the rotation matrix R"
        rotated_points = R @ points
        return rotated_points

    def translate_points(self, points, translation):
        "Translate points by the vector translation"
        translated_points = points + np.dot(translation,
                                            np.ones([1, points.shape[1]]))
        return translated_points

    def get_points(self):
        """"
            Points that define the spacecraft, and the colors of the triangular mesh
            Define the points on the spacecraft following information in Appendix C.3
        """
        w = 0.5  # Head width/2
        d = 0.5  # Head depth/2
        h = 1.0  # head height
        hb = 3.0  # body height
        db = 0.8  # body depth/2
        wb = 1.0  # body width/2
        hbeak = h * 0.4  # dist from neck to beak point
        wbeak = 0.2  # width of beak/2
        wl = hb * 0.15  # dist to top of white belly
        lb = 0.4  # length of beak
        lw = 3.0  # wing length from body center
        ww = 1.25  # height of wing in body center
        lf = 0.5  # length of feet
        wf = 0.5  # width of feet

        # points are in XYZ coordinates
        #   define the points on the spacecraft according to Appendix C.3
        points = np.array([[h, w, -d], [h, w, d], [h, -w, d], [h, -w, -d],
                           [0, w, -d], [0, w, d], [0, -w, d], [0, -w, -d],
                           [-hb, wb, -db], [-hb, wb, db], [-hb, -wb, db],
                           [-hb, -wb, -db], [-wl, 0, d + (db - d) * wl / hb],
                           [-wl, 0, 0], [-wl - ww, -lw, 0], [-wl - ww, 0, 0],
                           [-wl - ww, lw, 0], [hbeak, wbeak, d],
                           [hbeak, 0, d + lb], [hbeak, -wbeak, d],
                           [-hb, wf, 0], [-hb - lf, wf / 2, 0], [-hb, 0, 0],
                           [-hb - lf, -wf / 2, 0], [-hb, -wf, 0]]).T

        # scale points for better rendering
        scale = 10
        points = scale * points

        #   define the colors for each face of triangular mesh
        black = np.array([0., 26. / 255., 51. / 255., 1])
        white = np.array([1., 1., 1., 1])
        orange = np.array([1., 153. / 255., 0., 1])
        meshColors = np.empty((27, 3, 4), dtype=np.float32)
        meshColors[:23, :, :] = black
        meshColors[23, :, :] = white
        meshColors[24:, :, :] = orange
        return points, meshColors

    def points_to_mesh(self, points):
        """"
        Converts points to triangular mesh
        Each mesh face is defined by three 3D points
          (a rectangle requires two triangular mesh faces)
        """
        points = points.T
        mesh = np.array([[points[0], points[1], points[2]],
                         [points[0], points[2], points[3]],
                         [points[0], points[1], points[5]],
                         [points[0], points[4], points[5]],
                         [points[0], points[4], points[7]],
                         [points[0], points[7], points[3]],
                         [points[2], points[7], points[6]],
                         [points[2], points[7], points[3]],
                         [points[4], points[8], points[5]],
                         [points[8], points[5], points[9]],
                         [points[5], points[9], points[12]],
                         [points[5], points[12], points[6]],
                         [points[6], points[12], points[10]],
                         [points[6], points[11], points[7]],
                         [points[6], points[11], points[10]],
                         [points[8], points[10], points[9]],
                         [points[8], points[10], points[11]],
                         [points[13], points[16], points[15]],
                         [points[13], points[14], points[15]],
                         [points[4], points[8], points[7]],
                         [points[7], points[8], points[11]],
                         [points[1], points[2], points[5]],
                         [points[2], points[5], points[6]],
                         [points[9], points[12], points[10]],
                         [points[17], points[18], points[19]],
                         [points[20], points[21], points[22]],
                         [points[22], points[23], points[24]]])
        return mesh
