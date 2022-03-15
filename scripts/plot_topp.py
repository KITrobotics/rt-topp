#! /usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import numpy as np
from utils import (
    setup_matplotlib,
    path_parameterization_from_json,
    PathParameterization,
)


def plot_2d_path_parameterization(path_parameterization: PathParameterization):
    rows = 2
    _, ax = plt.subplots(nrows=rows, ncols=1, sharex=True, sharey=False)
    ax[0].set_title(r"Path velocity")
    ax[0].set_ylabel(r"$\dot{s}$")
    ax[1].set_title(r"Path acceleration")
    ax[1].set_ylabel(r"$\ddot{s}$")
    plt.xlabel(r"idx")

    ax[0].plot(
        range(path_parameterization.fw_path.s.reshape(-1).shape[0]),
        path_parameterization.fw_path.s_dot,
        label="fw param",
        marker=".",
    )
    ax[0].plot(
        range(path_parameterization.fw_path.s.reshape(-1).shape[0]),
        path_parameterization.limits.vel_abs_max,
        label="first + second max",
    )
    ax[0].plot(
        range(path_parameterization.bw_path.s.reshape(-1).shape[0]),
        path_parameterization.bw_path.s_dot,
        label="bw param",
        marker=".",
    )
    ax[0].set_ylim(
        0.0,
        np.mean(path_parameterization.limits.vel_abs_max) * 3.0,
    )

    ax[1].plot(
        range(path_parameterization.fw_path.s.reshape(-1).shape[0]),
        path_parameterization.fw_path.s_ddot,
        label="fw param",
        marker=".",
    )
    ax[1].plot(
        range(path_parameterization.fw_path.s.reshape(-1).shape[0]),
        path_parameterization.limits.acc_min,
        label="dyn min",
    )
    ax[1].plot(
        range(path_parameterization.fw_path.s.reshape(-1).shape[0]),
        path_parameterization.limits.acc_max,
        label="dyn max",
    )
    ax[1].plot(
        range(path_parameterization.fw_path.s.reshape(-1).shape[0]),
        path_parameterization.limits.backward_acc_min,
        label="bw dyn min",
    )
    ax[1].plot(
        range(path_parameterization.fw_path.s.reshape(-1).shape[0]),
        path_parameterization.limits.backward_acc_max,
        label="bw dyn max",
    )
    ax[1].plot(
        range(path_parameterization.bw_path.s.reshape(-1).shape[0]),
        path_parameterization.bw_path.s_ddot,
        label="bw param",
        marker=".",
    )
    ax[1].set_ylim(
        # -abs(np.mean(path_parameterization.fw_path.s_ddot)),
        # abs(np.mean(path_parameterization.fw_path.s_ddot)),
        np.min(path_parameterization.fw_path.s_ddot),
        np.max(path_parameterization.fw_path.s_ddot),
    )

    for i in range(rows):
        ax[i].grid()
        ax[i].legend()
        ax[i].set_xticks(
            range(0, path_parameterization.fw_path.s.reshape(-1).shape[0] + 1, 10)
        )
    plt.xlim(0, path_parameterization.fw_path.s.reshape(-1).shape[0])
    plt.tight_layout(pad=0.0)
    plt.show()


def plot_joint_parameterization(path_parameterization: PathParameterization):
    rows = 3
    n_joints = path_parameterization.joint_trajectory.velocities.shape[0]
    _, ax = plt.subplots(nrows=rows, ncols=1, sharex=True, sharey=False)
    ax[0].set_title(r"Joint positions")
    ax[0].set_ylabel(r"$\bm{q}$ [\si{\radian}]")
    ax[1].set_title(r"Joint velocities")
    ax[1].set_ylabel(r"$\bm{\dot{q}}$ [\si{\radian/\second}]")
    ax[2].set_title(r"Joint accelerations")
    ax[2].set_ylabel(r"$\bm{\ddot{q}}$ [\si{\radian/\square\second}]")
    plt.xlabel(r"idx")

    for i in range(n_joints):
        ax[0].plot(
            range(path_parameterization.bw_path.s.reshape(-1).shape[0]),
            path_parameterization.joint_trajectory.positions[i],
            marker=".",
            label=r"$q_" + str(i + 1) + "$",
        )
        ax[1].plot(
            range(path_parameterization.bw_path.s.reshape(-1).shape[0]),
            path_parameterization.joint_trajectory.velocities[i],
            marker=".",
            label=r"$\dot{q}_" + str(i + 1) + "$",
        )
        ax[2].plot(
            range(path_parameterization.bw_path.s.reshape(-1).shape[0]),
            path_parameterization.joint_trajectory.accelerations[i],
            marker=".",
            label=r"$\ddot{q}_" + str(i + 1) + "$",
        )
    dof_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][0:n_joints]
    ax[1].hlines(
        path_parameterization.limits.joint_vel_min,
        0,
        path_parameterization.fw_path.s.reshape(-1).shape[0],
        colors=dof_colors,
        linestyles="dashed",
        alpha=0.7,
    )
    ax[1].hlines(
        path_parameterization.limits.joint_vel_max,
        0,
        path_parameterization.fw_path.s.reshape(-1).shape[0],
        colors=dof_colors,
        linestyles="dashed",
        alpha=0.7,
    )
    ax[1].set_ylim(
        np.min(path_parameterization.limits.joint_vel_min) * 1.5,
        np.max(path_parameterization.limits.joint_vel_max) * 1.5,
    )

    ax[2].hlines(
        # TODO(wolfgang): assumes constant accelerations, change for dynamics!
        # if complete array is taken, plot becomes very slow
        path_parameterization.limits.joint_acc_min[:, 0],
        0,
        path_parameterization.fw_path.s.reshape(-1).shape[0],
        colors=dof_colors,
        linestyles="dashed",
        alpha=0.7,
    )
    ax[2].hlines(
        path_parameterization.limits.joint_acc_max[:, 0],
        0,
        path_parameterization.fw_path.s.reshape(-1).shape[0],
        colors=dof_colors,
        linestyles="dashed",
        alpha=0.7,
    )
    ax[2].set_ylim(
        np.min(path_parameterization.limits.joint_acc_min) * 1.5,
        np.max(path_parameterization.limits.joint_acc_max) * 1.5,
    )

    for i in range(rows):
        ax[i].grid()
        ax[i].legend()
        ax[i].set_xticks(
            range(0, path_parameterization.fw_path.s.reshape(-1).shape[0] + 1, 10)
        )
    plt.xlim(0, path_parameterization.fw_path.s.reshape(-1).shape[0])
    plt.tight_layout(pad=0.0)
    plt.show()


def plot_spline_data(path_parameterization: PathParameterization):
    n_joints = path_parameterization.joint_trajectory.velocities.shape[0]
    rows = 2
    _, ax = plt.subplots(nrows=rows, ncols=1, sharex=True, sharey=False)
    ax[0].set_title(r"Path position")
    ax[0].set_ylabel(r"$s$")
    ax[1].set_title(r"Spline derivatives")

    ax[0].plot(
        range(path_parameterization.bw_path.s.reshape(-1).shape[0]),
        path_parameterization.fw_path.s,
        label="fw positions",
        marker=".",
    )
    ax[0].plot(
        range(path_parameterization.bw_path.s.reshape(-1).shape[0]),
        path_parameterization.bw_path.s,
        label="bw positions",
        marker=".",
    )

    for i in range(n_joints):
        ax[1].plot(
            range(path_parameterization.bw_path.s.reshape(-1).shape[0]),
            path_parameterization.derivatives.first[i],
            marker=".",
            label=r"$q_" + str(i + 1) + "'$",
        )
        ax[1].plot(
            range(path_parameterization.bw_path.s.reshape(-1).shape[0]),
            path_parameterization.derivatives.second[i],
            marker=".",
            label=r"$q_" + str(i + 1) + "''$",
        )
        # ax[1].plot(
        #     range(path_parameterization.bw_path.s.reshape(-1).shape[0]),
        #     path_parameterization.derivatives.third[i],
        #     marker=".",
        #     label=r"$q_" + str(i + 1) + "'''$",
        # )

    for i in range(rows):
        ax[i].grid()
        ax[i].legend()
        ax[i].set_xticks(
            range(0, path_parameterization.fw_path.s.reshape(-1).shape[0] + 1, 10)
        )
    plt.xlim(0, path_parameterization.fw_path.s.reshape(-1).shape[0])
    plt.tight_layout(pad=0.0)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot topp for json trajectory")
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        required=True,
        help="file path",
    )
    parser.add_argument(
        "-j",
        "--joint",
        action="store_true",
        help="plot joint space",
    )
    parser.add_argument(
        "-s",
        "--spline",
        action="store_true",
        help="plot joint space",
    )
    args = parser.parse_args()
    json_path = args.path

    path_parameterization = path_parameterization_from_json(json_path)

    setup_matplotlib()

    if args.joint:
        plot_joint_parameterization(path_parameterization)
    elif args.spline:
        plot_spline_data(path_parameterization)
    else:
        plot_2d_path_parameterization(path_parameterization)

    # plotting EEF path would also be nice (needs kinematics operations)


if __name__ == "__main__":
    main()
