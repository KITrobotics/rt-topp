import json

import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


@dataclass(unsafe_hash=True)
class JointState:
    """
    Representation of joint angles and its time derivatives
    """

    angle: np.array
    velocity: np.array
    acceleration: np.array
    jerk: np.array


def dict_to_jointstate(data):
    jerk = []

    ang = data["values"]["angle"]
    vel = data["values"]["velocity"]
    acc = data["values"]["acceleration"]
    if "jerk" in data["values"]:
        jerk = data["values"]["jerk"]

    return JointState(
        np.array(ang).reshape(-1, 1),
        np.array(vel).reshape(-1, 1),
        np.array(acc).reshape(-1, 1),
        np.array(jerk).reshape(-1, 1),
    )


@dataclass(unsafe_hash=True)
class JointTrajectory:
    timestamps: np.array
    positions: np.array
    velocities: np.array
    accelerations: np.array
    jerks: np.array


def calculate_joint_trajectory(timestamps, joint_states) -> JointTrajectory:
    timestamps = np.array(timestamps).reshape(-1, 1)
    n_joints = joint_states[0].angle.shape[0]
    positions = []
    velocities = []
    accelerations = []
    jerks = []

    for state in joint_states:
        positions.append(state.angle)
        velocities.append(state.velocity)
        accelerations.append(state.acceleration)
        jerks.append(state.jerk)

    return JointTrajectory(
        timestamps,
        np.array(positions).reshape(-1, n_joints).T,
        np.array(velocities).reshape(-1, n_joints).T,
        np.array(accelerations).reshape(-1, n_joints).T,
        np.array(jerks).reshape(-1, n_joints).T,
    )


def jointtraj_from_json_block(data) -> JointTrajectory:
    joint_timestamps = []
    joint_states = []

    for p in data["jointtrajpoints"]:
        if "timestamp" in p:
            timestamp = p["timestamp"]
            joint_timestamps.append(timestamp)

        joint_state = dict_to_jointstate(p)
        joint_states.append(joint_state)

    return calculate_joint_trajectory(joint_timestamps, joint_states)


def setup_matplotlib():
    """
    Setup latex configuration to generate print quality plots with matplotlib
    """
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams["font.size"] = 14
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["text.usetex"] = True
    plt.rcParams[
        "text.latex.preamble"
    ] = r"\usepackage{siunitx} \usepackage{amsmath} \usepackage{bm}"
    plt.rcParams["pgf.preamble"] = plt.rcParams["text.latex.preamble"]
    plt.rcParams["legend.loc"] = "upper right"
    # plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["axes.titlesize"] = "large"
    plt.rcParams["axes.labelsize"] = "large"
    # plt.rcParams["xtick.labelsize"] = 12
    # plt.rcParams["ytick.labelsize"] = 12

    plt.rcParams["figure.autolayout"] = True


def set_aspect_equal_3d(ax):
    """
    Fix aspect ratio of a 3D plot
    kudos to https://stackoverflow.com/a/35126679
    :param ax: matplotlib 3D axes object
    """
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    from numpy import mean

    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)

    plot_radius = max(
        [
            abs(lim - mean_)
            for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean))
            for lim in lims
        ]
    )

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])


@dataclass(unsafe_hash=True)
class Path:
    s: np.array
    s_dot: np.array
    s_ddot: np.array
    s_dddot: np.array


def calculate_path(positions, velocities, accelerations, jerks) -> Path:
    position = np.array(positions).reshape(-1, 1)
    velocity = np.array(velocities).reshape(-1, 1)
    acceleration = np.array(accelerations).reshape(-1, 1)
    jerk = np.array(jerks).reshape(-1, 1)

    return Path(position, velocity, acceleration, jerk)


@dataclass(unsafe_hash=True)
class PathParameterizationLimits:
    forward_vel_abs_max: np.array
    backward_vel_abs_max: np.array
    acc_abs_min: np.array
    acc_abs_max: np.array
    jerk_abs_min: np.array
    jerk_abs_max: np.array

    acc_min: np.array
    acc_max: np.array
    backward_acc_min: np.array
    backward_acc_max: np.array
    jerk_min: np.array
    jerk_max: np.array

    singular_acceleration: np.array
    singular_jerk: np.array

    vel_third_max: np.array

    joint_vel_min: np.array
    joint_vel_max: np.array
    joint_acc_min: np.array
    joint_acc_max: np.array


def calculate_path_parameterization_limits(
    forward_vel_abs_max,
    backward_vel_abs_max,
    acc_abs_min,
    acc_abs_max,
    jerk_abs_min,
    jerk_abs_max,
    acc_min,
    acc_max,
    backward_acc_min,
    backward_acc_max,
    jerk_min,
    jerk_max,
    singular_acceleration,
    singular_jerk,
    vel_third_max,
    joint_vel_min,
    joint_vel_max,
    joint_acc_min,
    joint_acc_max,
) -> PathParameterizationLimits:
    forward_vel_abs_max = np.array(forward_vel_abs_max).reshape(-1, 1)
    backward_vel_abs_max = np.array(backward_vel_abs_max).reshape(-1, 1)
    acc_abs_min = np.array(acc_abs_min).reshape(-1, 1)
    acc_abs_max = np.array(acc_abs_max).reshape(-1, 1)
    jerk_abs_min = np.array(jerk_abs_min).reshape(-1, 1)
    jerk_abs_max = np.array(jerk_abs_max).reshape(-1, 1)

    acc_min = np.array(acc_min).reshape(-1, 1)
    acc_max = np.array(acc_max).reshape(-1, 1)
    backward_acc_min = np.array(backward_acc_min).reshape(-1, 1)
    backward_acc_max = np.array(backward_acc_max).reshape(-1, 1)
    jerk_min = np.array(jerk_min).reshape(-1, 1)
    jerk_max = np.array(jerk_max).reshape(-1, 1)

    singular_acceleration = np.array(singular_acceleration).reshape(-1, 1)
    singular_jerk = np.array(singular_jerk).reshape(-1, 1)

    vel_third_max = np.array(vel_third_max).reshape(-1, 1)

    n_joints = len(joint_acc_min[0])
    joint_vel_min = np.array(joint_vel_min).reshape(-1, n_joints).T
    joint_vel_max = np.array(joint_vel_max).reshape(-1, n_joints).T
    joint_acc_min = np.array(joint_acc_min).reshape(-1, n_joints).T
    joint_acc_max = np.array(joint_acc_max).reshape(-1, n_joints).T

    return PathParameterizationLimits(
        forward_vel_abs_max,
        backward_vel_abs_max,
        acc_abs_min,
        acc_abs_max,
        jerk_abs_min,
        jerk_abs_max,
        acc_min,
        acc_max,
        backward_acc_min,
        backward_acc_max,
        jerk_min,
        jerk_max,
        singular_acceleration,
        singular_jerk,
        vel_third_max,
        joint_vel_min,
        joint_vel_max,
        joint_acc_min,
        joint_acc_max,
    )


@dataclass(unsafe_hash=True)
class Derivatives:
    forward_first: np.array
    forward_second: np.array
    forward_third: np.array
    backward_first: np.array
    backward_second: np.array
    backward_third: np.array


def calculate_spline_derivatives(
    forward_first,
    forward_second,
    forward_third,
    backward_first,
    backward_second,
    backward_third,
) -> Derivatives:
    n_joints = len(forward_first[0])
    forward_firsts = np.array(forward_first).reshape(-1, n_joints).T
    forward_seconds = np.array(forward_second).reshape(-1, n_joints).T
    forward_thirds = np.array(forward_third).reshape(-1, n_joints).T
    backward_firsts = np.array(backward_first).reshape(-1, n_joints).T
    backward_seconds = np.array(backward_second).reshape(-1, n_joints).T
    backward_thirds = np.array(backward_third).reshape(-1, n_joints).T

    return Derivatives(
        forward_firsts,
        forward_seconds,
        forward_thirds,
        backward_firsts,
        backward_seconds,
        backward_thirds,
    )


@dataclass(unsafe_hash=True)
class PathParameterization:
    timestamps: np.array
    waypoint_indices: np.array
    waypoint_s: np.array
    fw_path: Path
    bw_path: Path
    limits: PathParameterizationLimits
    joint_trajectory: JointTrajectory
    optimization_time: float
    derivatives: Derivatives


def read_path_parameterization_from_json_block(json_block) -> PathParameterization:
    timestamps = []
    waypoint_indices = []
    waypoint_s = []
    positions = []
    forward_velocities = []
    forward_accelerations = []
    forward_jerks = []
    backward_velocities = []
    backward_accelerations = []
    backward_jerks = []

    vel_abs_max = []
    acc_abs_min = []
    acc_abs_max = []
    jerk_abs_min = []
    jerk_abs_max = []

    acc_min = []
    acc_max = []
    backward_acc_min = []
    backward_acc_max = []
    jerk_min = []
    jerk_max = []

    joint_vel_min = []
    joint_vel_max = []
    joint_acc_min = []
    joint_acc_max = []

    singular_acceleration = []
    singular_jerk = []

    vel_third_max = []

    derivatives_first = []
    derivatives_second = []
    derivatives_third = []

    joint_traj_points = {}
    joint_traj_points["jointtrajpoints"] = []

    for p in json_block["path_parameterization"]:
        if "timestamp" in p:
            timestamps.append(p["timestamp"])
        positions.append(p["position"])

        forward_velocities.append(p["forward"]["velocity"])
        forward_accelerations.append(p["forward"]["acceleration"])
        if "jerk" in p["forward"]:
            forward_jerks.append(p["forward"]["jerk"])

        backward_velocities.append(p["backward"]["velocity"])
        backward_accelerations.append(p["backward"]["acceleration"])
        if "jerk" in p["backward"]:
            backward_jerks.append(p["backward"]["jerk"])

        vel_abs_max.append(p["vel_abs_max"])
        if "acc_abs_min" in p:
            acc_abs_min.append(p["acc_abs_min"])
            acc_abs_max.append(p["acc_abs_max"])
            jerk_abs_min.append(p["jerk_abs_min"])
            jerk_abs_max.append(p["jerk_abs_max"])

        acc_min.append(p["acc_min"])
        acc_max.append(p["acc_max"])
        if "acc_min" in p["backward"]:
            backward_acc_min.append(p["backward"]["acc_min"])
            backward_acc_max.append(p["backward"]["acc_max"])
        if "jerk_min" in p:
            jerk_min.append(p["jerk_min"])
            jerk_max.append(p["jerk_max"])

            singular_acceleration.append(p["singular_acceleration"])
            singular_jerk.append(p["singular_jerk"])

            vel_third_max.append(p["vel_third_max"])

        joint_traj_points["jointtrajpoints"].append(p["jointtrajpoints"])

        joint_acc_min.append(p["joint_constraints"]["acc_min"])
        joint_acc_max.append(p["joint_constraints"]["acc_max"])

        if "derivatives" in p:
            derivatives = p["derivatives"]
            derivatives_first.append(derivatives["first"])
            derivatives_second.append(derivatives["second"])
            derivatives_third.append(derivatives["third"])

    if "joint_constraints" in json_block:
        joint_constraints = json_block["joint_constraints"]
        joint_vel_min = joint_constraints["vel_min"]
        joint_vel_max = joint_constraints["vel_max"]

    forward_path = calculate_path(
        positions, forward_velocities, forward_accelerations, forward_jerks
    )
    backward_path = calculate_path(
        positions, backward_velocities, backward_accelerations, backward_jerks
    )
    limits = calculate_path_parameterization_limits(
        vel_abs_max,
        vel_abs_max,  # put it twice cause we don't differentiate here
        acc_abs_min,
        acc_abs_max,
        jerk_abs_min,
        jerk_abs_max,
        acc_min,
        acc_max,
        backward_acc_min,
        backward_acc_max,
        jerk_min,
        jerk_max,
        singular_acceleration,
        singular_jerk,
        vel_third_max,
        joint_vel_min,
        joint_vel_max,
        joint_acc_min,
        joint_acc_max,
    )
    # put them twice, different values only needed for sampling
    derivatives = calculate_spline_derivatives(
        derivatives_first,
        derivatives_second,
        derivatives_third,
        derivatives_first,
        derivatives_second,
        derivatives_third,
    )

    joint_trajectory = jointtraj_from_json_block(joint_traj_points)
    optimization_time = 0.0
    if "path_parameterization_optimization_time" in json_block:
        optimization_time = (
            float(json_block["path_parameterization_optimization_time"]) / 1.0e6
        )

    if "waypoints" in json_block:
        for wp in json_block["waypoints"]:
            waypoint_indices.append(wp["idx_on_path"])

    if len(forward_path.s) > max(waypoint_indices):
        print(len(forward_path.s))
        print(waypoint_indices)
        waypoint_s = [forward_path.s[idx] for idx in waypoint_indices]

    return PathParameterization(
        np.array(timestamps),
        np.array(waypoint_indices, dtype=np.int16),
        np.array(waypoint_s),
        forward_path,
        backward_path,
        limits,
        joint_trajectory,
        optimization_time,
        derivatives,
    )


# TODO(wolfgang): refactor with the duplicate code above
def read_path_parameterization_from_json_block_sampling(
    json_block,
) -> PathParameterization:
    timestamps = []
    waypoint_indices = []
    waypoint_s = []
    forward_positions = []
    backward_positions = []
    forward_velocities = []
    forward_accelerations = []
    forward_jerks = []
    backward_velocities = []
    backward_accelerations = []
    backward_jerks = []

    forward_vel_abs_max = []
    backward_vel_abs_max = []
    acc_abs_min = []
    acc_abs_max = []
    jerk_abs_min = []
    jerk_abs_max = []

    acc_min = []
    acc_max = []
    backward_acc_min = []
    backward_acc_max = []
    jerk_min = []
    jerk_max = []

    joint_vel_min = []
    joint_vel_max = []
    joint_acc_min = []
    joint_acc_max = []

    singular_acceleration = []
    singular_jerk = []

    vel_third_max = []

    forward_derivatives_first = []
    forward_derivatives_second = []
    forward_derivatives_third = []
    backward_derivatives_first = []
    backward_derivatives_second = []
    backward_derivatives_third = []

    joint_traj_points = {}
    joint_traj_points["jointtrajpoints"] = []

    for p in json_block["path_parameterization_bw"]:
        backward_positions.append(p["position"])

        backward_velocities.append(p["velocity"])
        backward_accelerations.append(p["acceleration"])
        if "jerk" in p:
            backward_jerks.append(p["jerk"])

        backward_vel_abs_max.append(p["vel_abs_max"])
        if "acc_abs_min" in p:
            acc_abs_min.append(p["acc_abs_min"])
            acc_abs_max.append(p["acc_abs_max"])
            jerk_abs_min.append(p["jerk_abs_min"])
            jerk_abs_max.append(p["jerk_abs_max"])

        backward_acc_min.append(p["acc_min"])
        backward_acc_max.append(p["acc_max"])
        if "jerk_min" in p:
            jerk_min.append(p["jerk_min"])
            jerk_max.append(p["jerk_max"])

            singular_acceleration.append(p["singular_acceleration"])
            singular_jerk.append(p["singular_jerk"])

            vel_third_max.append(p["vel_third_max"])

        joint_acc_min.append(p["joint_constraints"]["acc_min"])
        joint_acc_max.append(p["joint_constraints"]["acc_max"])

        # TODO(wolfgang): get them separately for forward and backward? already supported by cpp code
        derivatives = p["derivatives"]
        backward_derivatives_first.append(derivatives["first"])
        backward_derivatives_second.append(derivatives["second"])
        backward_derivatives_third.append(derivatives["third"])

    for p in json_block["path_parameterization_fw"]:
        forward_positions.append(p["position"])

        forward_velocities.append(p["velocity"])
        forward_accelerations.append(p["acceleration"])

        acc_min.append(p["acc_min"])
        acc_max.append(p["acc_max"])
        forward_vel_abs_max.append(p["vel_abs_max"])

        timestamps.append(p["time"])

        joint_traj_points["jointtrajpoints"].append(p["jointtrajpoints"])

        derivatives = p["derivatives"]
        forward_derivatives_first.append(derivatives["first"])
        forward_derivatives_second.append(derivatives["second"])
        forward_derivatives_third.append(derivatives["third"])

    if "joint_constraints" in json_block:
        joint_constraints = json_block["joint_constraints"]
        joint_vel_min = joint_constraints["vel_min"]
        joint_vel_max = joint_constraints["vel_max"]

    forward_path = calculate_path(
        forward_positions, forward_velocities, forward_accelerations, forward_jerks
    )
    backward_path = calculate_path(
        backward_positions, backward_velocities, backward_accelerations, backward_jerks
    )
    limits = calculate_path_parameterization_limits(
        forward_vel_abs_max,
        backward_vel_abs_max,
        acc_abs_min,
        acc_abs_max,
        jerk_abs_min,
        jerk_abs_max,
        acc_min,
        acc_max,
        backward_acc_min,
        backward_acc_max,
        jerk_min,
        jerk_max,
        singular_acceleration,
        singular_jerk,
        vel_third_max,
        joint_vel_min,
        joint_vel_max,
        joint_acc_min,
        joint_acc_max,
    )
    derivatives = calculate_spline_derivatives(
        forward_derivatives_first,
        forward_derivatives_second,
        forward_derivatives_third,
        backward_derivatives_first,
        backward_derivatives_second,
        backward_derivatives_third,
    )

    joint_trajectory = jointtraj_from_json_block(joint_traj_points)
    optimization_time = 0.0
    if "path_parameterization_optimization_time" in json_block:
        optimization_time = (
            float(json_block["path_parameterization_optimization_time"]) / 1.0e6
        )

    if "waypoints" in json_block:
        for wp in json_block["waypoints"]:
            waypoint_indices.append(wp["idx_on_path"])

        waypoint_s = [backward_path.s[idx] for idx in waypoint_indices]

    return PathParameterization(
        np.array(timestamps),
        np.array(waypoint_indices, dtype=np.int16),
        np.array(waypoint_s),
        forward_path,
        backward_path,
        limits,
        joint_trajectory,
        optimization_time,
        derivatives,
    )


def path_parameterization_from_json(file_path, sampling=False) -> PathParameterization:
    with open(file_path) as json_file:
        data = json.load(json_file)

        if not sampling:
            return read_path_parameterization_from_json_block(data)

        return read_path_parameterization_from_json_block_sampling(data)
