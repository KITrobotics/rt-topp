#! /usr/bin/env python3

import argparse
from cProfile import label
from turtle import color

import matplotlib.pyplot as plt
import numpy as np
from utils import (
    setup_matplotlib,
    path_parameterization_from_json,
    PathParameterization,
)


def plot_2d_path_parameterization(path_parameterization: PathParameterization):
    rows = 2
    n_idx = path_parameterization.bw_path.s.reshape(-1).shape[0]
    fig_x = n_idx // 30
    fig, ax = plt.subplots(
        nrows=rows, ncols=1, sharex=True, sharey=False, figsize=(fig_x, 8)
    )
    ax[0].set_title(r"Path velocity")
    ax[0].set_ylabel(r"$\dot{s} \quad [\mathrm{rad/s}]$")
    ax[1].set_title(r"Path acceleration")
    ax[1].set_ylabel(r"$\ddot{s} \quad [\mathrm{rad/s^2}]$")
    plt.xlabel(r"$s$")

    ax[0].plot(
        range(path_parameterization.fw_path.s.reshape(-1).shape[0]),
        path_parameterization.fw_path.s_dot,
        label="forward param",
        # marker=".",
    )
    ax[0].plot(
        range(path_parameterization.fw_path.s.reshape(-1).shape[0]),
        path_parameterization.limits.forward_vel_abs_max,
        label="MVC (max velocity)",
    )
    ax[0].plot(
        range(path_parameterization.bw_path.s.reshape(-1).shape[0]),
        path_parameterization.bw_path.s_dot,
        label="backward param",
        # marker=".",
    )
    ax[0].vlines(
        x=path_parameterization.waypoint_indices,
        ymax=np.zeros_like(path_parameterization.waypoint_indices),
        ymin=np.ones_like(path_parameterization.waypoint_indices)
        * np.mean(path_parameterization.limits.forward_vel_abs_max)
        * 3.0,
        label="waypoints",
        color="b",
    )
    ax[0].set_ylim(
        0.0,
        np.mean(path_parameterization.limits.forward_vel_abs_max) * 3.0,
    )

    ax[1].plot(
        range(path_parameterization.fw_path.s.reshape(-1).shape[0]),
        path_parameterization.fw_path.s_ddot,
        label="fw param",
        # marker=".",
    )
    ax[1].plot(
        range(path_parameterization.fw_path.s.reshape(-1).shape[0]),
        path_parameterization.limits.acc_min,
        label="$a_{min}$",
    )
    ax[1].plot(
        range(path_parameterization.fw_path.s.reshape(-1).shape[0]),
        path_parameterization.limits.acc_max,
        label="$a_{max}$",
    )
    # ax[1].plot(
    #     range(path_parameterization.fw_path.s.reshape(-1).shape[0]),
    #     path_parameterization.limits.backward_acc_min,
    #     label="bw dyn min",
    # )
    # ax[1].plot(
    #     range(path_parameterization.fw_path.s.reshape(-1).shape[0]),
    #     path_parameterization.limits.backward_acc_max,
    #     label="bw dyn max",
    # )
    ax[1].plot(
        range(path_parameterization.bw_path.s.reshape(-1).shape[0]),
        path_parameterization.bw_path.s_ddot,
        label="bw param",
        # marker=".",
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
            range(0, path_parameterization.fw_path.s.reshape(-1).shape[0] + 1, 50)
        )
    plt.xlim(0, path_parameterization.fw_path.s.reshape(-1).shape[0])

    fig.set_size_inches(w=6.0, h=5.0)
    plt.tight_layout(pad=0.0)
    plt.savefig("topp_phase.pgf", dpi=400)

    plt.show()


def plot_2d_path_velocity(path_parameterization: PathParameterization):
    rows = 1
    n_idx = path_parameterization.bw_path.s.reshape(-1).shape[0]
    fig_x = n_idx // 30
    fig, ax = plt.subplots(
        nrows=rows, ncols=1, sharex=True, sharey=False, figsize=(fig_x, 8)
    )
    ax.set_title(r"Path velocity")
    ax.set_ylabel(r"$\dot{s} \quad [\mathrm{rad/s}]$")
    plt.xlabel(r"$s$")

    ax.fill_between(
        range(path_parameterization.fw_path.s.reshape(-1).shape[0]),
        np.zeros_like(path_parameterization.limits.forward_vel_abs_max).flatten(),
        path_parameterization.limits.forward_vel_abs_max.flatten(),
        color="green",
        alpha=0.2,
        label="valid range",
    )

    ax.fill_between(
        range(path_parameterization.fw_path.s.reshape(-1).shape[0]),
        path_parameterization.limits.forward_vel_abs_max.flatten(),
        20.0,
        color="red",
        alpha=0.2,
        label="forbidden range",
    )

    ax.plot(
        range(path_parameterization.fw_path.s.reshape(-1).shape[0]),
        path_parameterization.fw_path.s_dot,
        label="forward param",
        marker=".",
    )
    ax.plot(
        range(path_parameterization.fw_path.s.reshape(-1).shape[0]),
        path_parameterization.limits.forward_vel_abs_max,
        label="MVC (max velocity)",
    )
    ax.plot(
        range(path_parameterization.bw_path.s.reshape(-1).shape[0]),
        path_parameterization.bw_path.s_dot,
        label="backward param",
        marker=".",
    )
    ax.vlines(
        x=path_parameterization.waypoint_indices,
        ymax=np.zeros_like(path_parameterization.waypoint_indices),
        ymin=np.ones_like(path_parameterization.waypoint_indices)
        * np.mean(path_parameterization.limits.forward_vel_abs_max)
        * 3.0,
        label="waypoints",
        color="b",
    )
    ax.set_ylim(
        0.0,
        np.mean(path_parameterization.limits.forward_vel_abs_max) * 2.2,
    )

    ax.grid()
    ax.legend()
    ax.set_xticks(
        range(0, path_parameterization.fw_path.s.reshape(-1).shape[0] + 1, 50)
    )
    plt.xlim(0, path_parameterization.fw_path.s.reshape(-1).shape[0])

    # data: -p ../data/random_waypoints_generic_sampling/param_random_waypoints_full_5_0.json
    # https://jwalton.info/Embed-Publication-Matplotlib-Latex/#comment-4904502735
    # size for thesis
    fig.set_size_inches(w=6.0, h=4.5)
    plt.tight_layout(pad=0.0)
    plt.savefig("topp_vel_phase.pgf", dpi=400)

    plt.show()


def plot_2d_path_parameterization_sampling(path_parameterization: PathParameterization):
    rows = 2
    n_idx = path_parameterization.bw_path.s.reshape(-1).shape[0]
    fig_x = n_idx // 30
    _, ax = plt.subplots(
        nrows=rows, ncols=1, sharex=True, sharey=False, figsize=(fig_x, 8)
    )
    ax[0].set_title(r"Path velocity")
    ax[0].set_ylabel(r"$\dot{s}$")
    ax[1].set_title(r"Path acceleration")
    ax[1].set_ylabel(r"$\ddot{s}$")
    plt.xlabel(r"s")

    ax[0].plot(
        path_parameterization.fw_path.s,
        path_parameterization.fw_path.s_dot,
        label="fw param",
        marker=".",
    )
    ax[0].plot(
        path_parameterization.fw_path.s,
        path_parameterization.limits.forward_vel_abs_max,
        label="fw first + second max",
    )
    ax[0].plot(
        path_parameterization.bw_path.s,
        path_parameterization.limits.backward_vel_abs_max,
        label="bw first + second max",
        marker=".",
    )
    ax[0].plot(
        path_parameterization.bw_path.s,
        path_parameterization.bw_path.s_dot,
        label="bw param",
        marker=".",
    )
    # TODO(wolfgang): add this again by using s position instead of indices
    # ax[0].vlines(
    #     x=path_parameterization.waypoint_indices,
    #     ymax=np.zeros_like(path_parameterization.waypoint_indices),
    #     ymin=np.ones_like(path_parameterization.waypoint_indices)
    #     * np.mean(path_parameterization.limits.vel_abs_max)
    #     * 3.0,
    #     label="Waypoints",
    #     color="b",
    # )
    ax[0].set_ylim(
        0.0,
        np.mean(path_parameterization.limits.backward_vel_abs_max) * 3.0,
    )

    ax[1].plot(
        path_parameterization.fw_path.s,
        path_parameterization.fw_path.s_ddot,
        label="fw param",
        marker=".",
    )
    ax[1].plot(
        path_parameterization.fw_path.s,
        path_parameterization.limits.acc_min,
        label="dyn min",
    )
    ax[1].plot(
        path_parameterization.fw_path.s,
        path_parameterization.limits.acc_max,
        label="dyn max",
    )
    ax[1].plot(
        path_parameterization.bw_path.s,
        path_parameterization.limits.backward_acc_min,
        label="bw dyn min",
    )
    ax[1].plot(
        path_parameterization.bw_path.s,
        path_parameterization.limits.backward_acc_max,
        label="bw dyn max",
    )
    ax[1].plot(
        path_parameterization.bw_path.s,
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
    plt.xlim(0, path_parameterization.fw_path.s[-1])
    plt.tight_layout(pad=0.0)
    plt.show()


def plot_joint_parameterization(path_parameterization: PathParameterization):
    rows = 3
    n_joints = path_parameterization.joint_trajectory.velocities.shape[0]
    n_idx = path_parameterization.bw_path.s.reshape(-1).shape[0]
    fig_x = n_idx // 30
    fig, ax = plt.subplots(
        nrows=rows, ncols=1, sharex=True, sharey=False, figsize=(fig_x, 8)
    )
    ax[0].set_title(r"Joint positions")
    ax[0].set_ylabel(r"$\bm{q}$ [\si{\radian}]")
    ax[1].set_title(r"Joint velocities")
    ax[1].set_ylabel(r"$\bm{\dot{q}}$ [\si{\radian/\second}]")
    ax[2].set_title(r"Joint accelerations")
    ax[2].set_ylabel(r"$\bm{\ddot{q}}$ [\si{\radian/\square\second}]")
    plt.xlabel(r"$s$")  # actually number of gridpoints, not rad

    ax[0].vlines(
        x=path_parameterization.waypoint_indices,
        ymax=np.ones_like(path_parameterization.waypoint_indices) * 3.14,
        ymin=np.ones_like(path_parameterization.waypoint_indices) * -3.14,
        label="waypoints",
        color="b",
    )

    # markers removed for smaller lines and file size pgf
    for i in range(n_joints):
        ax[0].plot(
            range(path_parameterization.bw_path.s.reshape(-1).shape[0]),
            path_parameterization.joint_trajectory.positions[i],
            # marker=".",
            label=r"$q_" + str(i + 1) + "$",
        )
        ax[1].plot(
            range(path_parameterization.bw_path.s.reshape(-1).shape[0]),
            path_parameterization.joint_trajectory.velocities[i],
            # marker=".",
            label=r"$\dot{q}_" + str(i + 1) + "$",
        )
        ax[2].plot(
            range(path_parameterization.bw_path.s.reshape(-1).shape[0]),
            path_parameterization.joint_trajectory.accelerations[i],
            # marker=".",
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
    ax[1].vlines(
        x=path_parameterization.waypoint_indices,
        ymax=np.ones_like(path_parameterization.waypoint_indices)
        * np.max(path_parameterization.limits.joint_vel_max)
        * 1.5,
        ymin=np.ones_like(path_parameterization.waypoint_indices)
        * np.min(path_parameterization.limits.joint_vel_min)
        * 1.5,
        label="waypoints",
        color="b",
    )
    ax[1].set_ylim(
        np.min(path_parameterization.limits.joint_vel_min) * 1.1,
        np.max(path_parameterization.limits.joint_vel_max) * 1.1,
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
        np.min(path_parameterization.limits.joint_acc_min) * 1.1,
        np.max(path_parameterization.limits.joint_acc_max) * 1.1,
    )
    ax[2].vlines(
        x=path_parameterization.waypoint_indices,
        ymax=np.ones_like(path_parameterization.waypoint_indices)
        * np.max(path_parameterization.limits.joint_acc_max)
        * 1.5,
        ymin=np.ones_like(path_parameterization.waypoint_indices)
        * np.min(path_parameterization.limits.joint_acc_min)
        * 1.5,
        label="waypoints",
        color="b",
    )

    for i in range(rows):
        ax[i].grid()
        ax[i].legend()
        ax[i].set_xticks(
            range(0, path_parameterization.fw_path.s.reshape(-1).shape[0] + 1, 50)
        )
    plt.xlim(0, path_parameterization.fw_path.s.reshape(-1).shape[0])
    plt.tight_layout(pad=0.0)

    # thesis
    fig.set_size_inches(w=6.0, h=7.5)
    plt.tight_layout(pad=0.0)
    plt.savefig("topp_joints.pgf", dpi=400)
    # plt.savefig("topp_joints.pdf", bbox_inches='tight')

    plt.show()


def plot_spline_data(path_parameterization: PathParameterization):
    n_joints = path_parameterization.joint_trajectory.velocities.shape[0]
    rows = 2
    n_idx = path_parameterization.bw_path.s.reshape(-1).shape[0]
    fig_x = n_idx // 30
    _, ax = plt.subplots(
        nrows=rows, ncols=1, sharex=True, sharey=False, figsize=(fig_x, 8)
    )
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
            path_parameterization.derivatives.backward_first[i],
            marker=".",
            label=r"$q_" + str(i + 1) + "'$",
        )
        ax[1].plot(
            range(path_parameterization.bw_path.s.reshape(-1).shape[0]),
            path_parameterization.derivatives.backward_second[i],
            marker=".",
            label=r"$q_" + str(i + 1) + "''$",
        )
        # ax[1].plot(
        #     range(path_parameterization.bw_path.s.reshape(-1).shape[0]),
        #     path_parameterization.derivatives.third[i],
        #     marker=".",
        #     label=r"$q_" + str(i + 1) + "'''$",
        # )

    ax[1].vlines(
        x=path_parameterization.waypoint_indices,
        ymax=np.ones_like(path_parameterization.waypoint_indices)
        * np.max(path_parameterization.derivatives.backward_second)
        * 1.5,
        ymin=np.ones_like(path_parameterization.waypoint_indices)
        * np.min(path_parameterization.derivatives.backward_second)
        * 1.5,
        label="Waypoints",
        color="b",
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


def plot_spline_data_sampling(path_parameterization: PathParameterization):
    n_joints = path_parameterization.joint_trajectory.velocities.shape[0]
    rows = 2
    n_idx = path_parameterization.bw_path.s.reshape(-1).shape[0]
    fig_x = n_idx // 30
    _, ax = plt.subplots(
        nrows=rows, ncols=1, sharex=True, sharey=False, figsize=(fig_x, 8)
    )
    ax[0].set_title(r"First derivatives")
    ax[1].set_title(r"Second derivatives")
    plt.xlabel(r"s")

    for i in range(n_joints):
        ax[0].plot(
            path_parameterization.fw_path.s,
            path_parameterization.derivatives.forward_first[i],
            marker=".",
            label=r"forward $q_" + str(i + 1) + "'$",
        )
    for i in range(n_joints):
        ax[0].plot(
            path_parameterization.bw_path.s,
            path_parameterization.derivatives.backward_first[i],
            marker=".",
            label=r"backward $q_" + str(i + 1) + "'$",
        )
    ax[0].vlines(
        x=path_parameterization.waypoint_s,
        ymax=np.ones_like(path_parameterization.waypoint_s)
        * np.max(path_parameterization.derivatives.backward_first)
        * 1.5,
        ymin=np.ones_like(path_parameterization.waypoint_s)
        * np.min(path_parameterization.derivatives.backward_first)
        * 1.5,
        label="Waypoints",
        color="b",
    )

    for i in range(n_joints):
        ax[1].plot(
            path_parameterization.fw_path.s,
            path_parameterization.derivatives.forward_second[i],
            marker=".",
            label=r"forward $q_" + str(i + 1) + "''$",
        )
    for i in range(n_joints):
        ax[1].plot(
            path_parameterization.bw_path.s,
            path_parameterization.derivatives.backward_second[i],
            marker=".",
            label=r"backward $q_" + str(i + 1) + "''$",
        )

    ax[1].vlines(
        x=path_parameterization.waypoint_s,
        ymax=np.ones_like(path_parameterization.waypoint_s)
        * np.max(path_parameterization.derivatives.backward_second)
        * 1.5,
        ymin=np.ones_like(path_parameterization.waypoint_s)
        * np.min(path_parameterization.derivatives.backward_second)
        * 1.5,
        label="Waypoints",
        color="b",
    )

    for i in range(rows):
        ax[i].grid()
        ax[i].legend()
    plt.xlim(0, path_parameterization.fw_path.s[-1])
    plt.tight_layout(pad=0.0)
    plt.show()


def plot_2d_interpolated_spline(
    path_parameterization: PathParameterization, joint_list: list
):
    n_visualize_joints = len(joint_list)
    n_dof = 2
    n_combinations = np.math.factorial(n_visualize_joints) / (
        np.math.factorial(n_visualize_joints - n_dof) * np.math.factorial(n_dof)
    )
    n_combinations = np.int(n_combinations)

    n_col = 3 if n_combinations <= 8 else 5
    n_rows = np.int(np.ceil(n_combinations / n_col))

    _, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_col,
        sharex=False,
        sharey=False,
        figsize=(4 * n_col, 4 * n_rows),
        dpi=80,
    )

    cnt = 0
    for i in range(n_visualize_joints):
        for j in range(i + 1, n_visualize_joints):
            if cnt >= n_combinations:
                break

            xx = joint_list[i]
            yy = joint_list[j]

            row = cnt // n_col
            col = cnt % n_col

            ax = axs[row, col] if n_combinations > n_col else axs[col]

            ax.plot(
                path_parameterization.joint_trajectory.positions[xx],
                path_parameterization.joint_trajectory.positions[yy],
                marker=".",
                label="Spline",
            )

            ax.plot(
                path_parameterization.joint_trajectory.positions[xx][0],
                path_parameterization.joint_trajectory.positions[yy][0],
                marker=".",
                markersize=15.0,
                color="r",
                label="Start",
            )

            ax.plot(
                path_parameterization.joint_trajectory.positions[xx][-1],
                path_parameterization.joint_trajectory.positions[yy][-1],
                marker=".",
                markersize=15.0,
                color="c",
                label="End",
            )

            wp_idx = path_parameterization.waypoint_indices
            ax.scatter(
                path_parameterization.joint_trajectory.positions[xx][wp_idx],
                path_parameterization.joint_trajectory.positions[yy][wp_idx],
                alpha=1,
                s=100,
                color="y",
                label="Waypoint",
            )

            max_x = max(path_parameterization.joint_trajectory.positions[xx])
            min_x = min(path_parameterization.joint_trajectory.positions[xx])

            max_y = max(path_parameterization.joint_trajectory.positions[yy])
            min_y = min(path_parameterization.joint_trajectory.positions[yy])

            max_limit = max(max_x, max_y)
            min_limit = min(min_x, min_y)

            scaling = 1.05
            max_limit = max(max_limit * scaling, max_limit * scaling * -1)
            min_limit = min(min_limit * scaling, min_limit * scaling * -1)

            ax.set_xlim(min_limit, max_limit)
            ax.set_ylim(min_limit, max_limit)

            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(r"$q_{}$".format(xx))
            ax.set_ylabel(r"$q_{}$".format(yy))
            cnt += 1

        if n_combinations > n_col:
            axs[0, 0].legend(loc=0, fontsize="x-small")
        else:
            axs[0].legend(loc=0, fontsize="x-small")

    plt.subplots_adjust(
        left=0, bottom=0.05, right=0.95, top=0.95, wspace=0, hspace=0.25
    )
    plt.suptitle("Projected spline in 2D")


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
    parser.add_argument(
        "-t",
        "--tsampling",
        action="store_true",
        help="plot time sampling data",
    )
    parser.add_argument(
        "-ps2",
        "--projectedspline2",
        nargs="*",  # 0 or more values expected => creates a list
        type=int,
        help="visualize the projected spline interpolation in 2D",
    )
    args = parser.parse_args()
    json_path = args.path

    path_parameterization = path_parameterization_from_json(json_path, args.tsampling)

    setup_matplotlib()

    if not args.tsampling and args.projectedspline2:
        plot_2d_interpolated_spline(path_parameterization, args.projectedspline2)

    if args.joint:
        plot_joint_parameterization(path_parameterization)
    elif args.spline and not args.tsampling:
        plot_spline_data(path_parameterization)
    elif args.spline and args.tsampling:
        plot_spline_data_sampling(path_parameterization)
    elif args.tsampling and not args.spline:
        plot_2d_path_parameterization_sampling(path_parameterization)
    else:
        plot_2d_path_parameterization(path_parameterization)
        # plot_2d_path_velocity(path_parameterization)

    # plotting EEF path would also be nice (needs kinematics operations)


if __name__ == "__main__":
    main()
