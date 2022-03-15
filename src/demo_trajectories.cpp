#include <rttopp/demo_trajectories.h>

namespace rttopp::demo_trajectories {

JointConstraints<NUM_IIWA_JOINTS> generateIIWAJointConstraints() {
  JointConstraints<NUM_IIWA_JOINTS> joint_constraints;

  joint_constraints.velocity_max << 1.7104, 1.7104, 1.7453, 2.2689, 2.4434,
      3.1415, 3.1415;
  joint_constraints.velocity_min = -joint_constraints.velocity_max;
  joint_constraints.acceleration_max << 5.4444, 5.4444, 5.5555, 7.2222, 7.7777,
      10.0, 10.0;
  joint_constraints.acceleration_min = -joint_constraints.acceleration_max;
  joint_constraints.jerk_max << 108.0, 108.0, 111.0, 144.0, 155.0, 200.0, 200.0;
  joint_constraints.jerk_min = -joint_constraints.jerk_max;

  joint_constraints.torque_max << 176.0, 176.0, 110.0, 110.0, 110.0, 40.0, 40.0;
  joint_constraints.torque_min = -joint_constraints.torque_max;

  return joint_constraints;
}

std::array<double, NUM_IIWA_JOINTS> iiwaJointPositionLimits() {
  return {2.93215, 2.05949, 2.93215, 2.05949, 2.93215, 2.05949, 3.01942};
}

}  // namespace rttopp::demo_trajectories
