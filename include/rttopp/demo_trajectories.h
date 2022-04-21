#pragma once

#include <random>

#include <rttopp/types_utils.h>

namespace rttopp::demo_trajectories {

const size_t NUM_IIWA_JOINTS = 7;

JointConstraints<NUM_IIWA_JOINTS> generateIIWAJointConstraints();
std::array<double, NUM_IIWA_JOINTS> iiwaJointPositionLimits();

template <std::size_t N_JOINTS>
JointConstraints<N_JOINTS> generateGenericJointConstraints() {
  JointConstraints<N_JOINTS> joint_constraints;

  joint_constraints.velocity_max =
      5.0 * WaypointJointDataType<N_JOINTS>::Ones();
  joint_constraints.velocity_min = -joint_constraints.velocity_max;
  joint_constraints.acceleration_max =
      10.0 * WaypointJointDataType<N_JOINTS>::Ones();
  joint_constraints.acceleration_min = -joint_constraints.acceleration_max;
  joint_constraints.jerk_max = 100.0 * WaypointJointDataType<N_JOINTS>::Ones();
  joint_constraints.jerk_min = -joint_constraints.jerk_max;

  joint_constraints.torque_max =
      200.0 * WaypointJointDataType<N_JOINTS>::Ones();
  joint_constraints.torque_min = -joint_constraints.torque_max;

  return joint_constraints;
}

template <std::size_t N_JOINTS>
std::array<double, N_JOINTS> genericJointPositionLimits() {
  std::array<double, N_JOINTS> limits;
  limits.fill(2.5);

  return limits;
}

template <std::size_t N_JOINTS>
Waypoints<N_JOINTS> generateRandomJointWaypoints(
    const std::size_t n_waypoints, std::array<double, N_JOINTS> limits) {
  Waypoints<N_JOINTS> waypoints;
  // TODO(wolfgang): 42 ok? (needs to be predictable)
  static std::mt19937 gen(42);  // NOLINT cert-msc32-c,cert-msc51-cpp
  std::array<std::uniform_real_distribution<>, N_JOINTS> joint_values;

  for (std::size_t i = 0; i < N_JOINTS; ++i) {
    joint_values[i] = std::uniform_real_distribution<>(-limits[i], limits[i]);
  }

  for (std::size_t n = 0; n < n_waypoints; ++n) {
    Waypoint<N_JOINTS> waypoint;
    for (size_t i = 0; i < N_JOINTS; ++i) {
      waypoint.joints.position[i] = joint_values[i](gen);
    }
    waypoints.push_back(waypoint);
  }

  return waypoints;
}

}  // namespace rttopp::demo_trajectories
