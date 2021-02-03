"""
mavsimPy
    - Chapter 3 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        12/18/2018 - RWB
        1/14/2019 - RWB
        2/24/2020 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
import parameters.simulation_parameters as SIM

from chap2.mav_viewer import mavViewer
# from chap3.data_viewer import dataViewer
from chap3.mav_dynamics import mavDynamics

# initialize the visualization
VIDEO = False  # True==write video, False==don't write video
mav_view = mavViewer()  # initialize the mav viewer
# data_view = dataViewer()  # initialize view of data plots
if VIDEO is True:
    from chap2.video_writer import videoWriter
    video = videoWriter(video_name="chap3_video.avi",
                        bounding_box=(0, 0, 1000, 1000),
                        output_rate=SIM.ts_video)

# initialize elements of the architecture
mav = mavDynamics(SIM.ts_simulation)

# initialize the simulation time
sim_time = SIM.start_time

# main simulation loop
print("Press Command-Q to exit...")
while sim_time < SIM.end_time:
    # -------vary forces and moments to check dynamics-------------
    fx = 10
    fy = 0  # 10
    fz = 0  # 10
    Mx = 0.1  # 0.1
    My = 0  # 0.1
    Mz = 0  # 0.1
    forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T

    # -------physical system-------------
    mav.update(forces_moments)  # propagate the MAV dynamics
    mav.true_state.north = mav.true_state.pn
    mav.true_state.east = mav.true_state.pe
    mav.true_state.altitude = mav.true_state.h

    # -------update viewer-------------
    mav_view.update(mav.true_state)  # plot body of MAV
    # data_view.update(
    #     mav.true_state,  # true states
    #     mav.true_state,  # estimated states
    #     mav.true_state,  # commanded states
        # SIM.ts_simulation)
    if VIDEO is True:
        video.update(sim_time)

    #-------increment time-------------
    sim_time += SIM.ts_simulation
    print("N: {:.2f}".format(mav.true_state.north), "\t", "E: {:.2f}".format(mav.true_state.east), "\t", "Alt: {:.2f}".format(mav.true_state.altitude), "\t", "Theta: {:.2f}".format(mav.true_state.theta), "\t", "Phi: {:.2f}".format(mav.true_state.phi), "\t", "Psi: {:.2f}".format(mav.true_state.psi))

if VIDEO is True:
    video.close()
