#include <iostream>
#include <sys/mman.h>  // mlockall()

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

JointConstraints<NUM_IIWA_JOINTS> generateAsymmetricJointConstraints() {
  JointConstraints<NUM_IIWA_JOINTS> joint_constraints;

  joint_constraints.velocity_max << 1.7104, 1.7104, 1.7453, 2.2689, 2.4434,
      3.1415, 3.1415;
  joint_constraints.velocity_min << -1.704, -0.504, -1.2453, -2.3689, -2.9434,
      -1.1415, -2.1415;
  ;
  joint_constraints.acceleration_max << 5.4444, 2.4444, 5.5555, 5.2222, 7.7777,
      9.0, 10.0;
  joint_constraints.acceleration_min << -2.4444, -5.4444, -5.5555, -7.2222,
      -7.7777, -10.0, -5.0;
  joint_constraints.jerk_max << 108.0, 108.0, 111.0, 144.0, 155.0, 200.0, 200.0;
  joint_constraints.jerk_min = -joint_constraints.jerk_max;

  joint_constraints.torque_max << 176.0, 176.0, 110.0, 110.0, 110.0, 40.0, 40.0;
  joint_constraints.torque_min = -joint_constraints.torque_max;

  return joint_constraints;
}

std::array<double, NUM_IIWA_JOINTS> iiwaJointPositionLimits() {
  return {2.93215, 2.05949, 2.93215, 2.05949, 2.93215, 2.05949, 3.01942};
}

void initPerf() {
  const int THREAD_PRIORITY = sched_get_priority_max(SCHED_FIFO);
  if (THREAD_PRIORITY == -1) {
    std::cout << "Error: unable to get maximum possible thread priority"
              << std::endl;
    return;
  }

  sched_param thread_param{};
  thread_param.sched_priority = THREAD_PRIORITY;
  if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &thread_param) != 0) {
    std::cout << "Error: unable to set realtime scheduling" << std::endl;
    return;
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  int cpu_core = 2;

  std::cout << "setting affinity to cpu core " << cpu_core << std::endl;
  CPU_SET(cpu_core, &cpuset);
  if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
    std::cout << "Error: unable to set affinity of realtime thread"
              << std::endl;
  }

  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    std::cout << "failed to lock memory" << std::endl;
    exit(EXIT_FAILURE);
  }
}

}  // namespace rttopp::demo_trajectories
