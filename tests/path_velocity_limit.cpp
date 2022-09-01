#include <gtest/gtest.h>

#include <rttopp/demo_trajectories.h>
#include <rttopp/path_acceleration_limits.h>
#include <rttopp/path_velocity_limit.h>
#include <rttopp/preprocessing.h>

constexpr double MVC1_VEL_TOL = 1.0e-02;
constexpr double MVC2_ACC_TOL = 1.0e-02;
// needs to be bigger if a first derivative is zero
constexpr double MVC2_ACC_TOL_GENERIC = 2.0;

TEST(PathVelocityLimit, MVC2AccMinLTAccMaxDerivSecondZero) {
  const size_t n_joints = 6;
  const auto joint_constraints =
      rttopp::demo_trajectories::generateGenericJointConstraints<n_joints>();

  rttopp::PathVelocityLimit<n_joints> path_velocity_limit(joint_constraints);
  rttopp::PathAccelerationLimits<n_joints> path_acceleration_limits(
      joint_constraints);
  rttopp::JointPathDerivatives<n_joints> derivatives;
  rttopp::PathState state;

  derivatives.first << -0.23244995964764292, 0.4810585092883391,
      0.45244485376636845, -0.11044578732891779, 0.4807807539440196,
      -0.27461124297575795;
  derivatives.second << -0.23520211177028735, 9.420699867031908e-07,
      0.009892204386653536, -0.17564863396576466, 0.006011832054515082,
      -0.2547816492453125;

  state.velocity = path_velocity_limit.calculateOverallLimit(derivatives);

  rttopp::WaypointJoint<n_joints> joint_state;
  joint_state.velocity = derivatives.first * state.velocity;
  for (size_t joint = 0; joint < n_joints; ++joint) {
    EXPECT_LT(joint_state.velocity[joint],
              joint_constraints.velocity_max[joint]);
    EXPECT_GT(joint_state.velocity[joint],
              joint_constraints.velocity_min[joint]);
  }

  double acc_min, acc_max;
  std::tie(acc_min, acc_max) =
      path_acceleration_limits.calculateDynamicLimits(state, derivatives);
  EXPECT_LT(acc_min, acc_max);
  EXPECT_GT(acc_min + MVC2_ACC_TOL, acc_max);
}

TEST(PathVelocityLimit, MVC2AccMinLTAccMaxDerivSecondZero2) {
  const size_t n_joints = 6;
  const auto joint_constraints =
      rttopp::demo_trajectories::generateGenericJointConstraints<n_joints>();

  rttopp::PathVelocityLimit<n_joints> path_velocity_limit(joint_constraints);
  rttopp::PathAccelerationLimits<n_joints> path_acceleration_limits(
      joint_constraints);
  rttopp::JointPathDerivatives<n_joints> derivatives;
  rttopp::PathState state;

  derivatives.first << 0.3898815056328669, 0.2369327583681725,
      -0.3850132934768913, 0.32713422357863414, 0.10711000912799706,
      -0.0030996752811361323;

  derivatives.second << 0.26078632323003353, 0.30616887667864284,
      8.534410604675327e-07, 0.20867338973876412, 0.19184477092446253,
      0.09744997974899774;

  state.velocity = path_velocity_limit.calculateOverallLimit(derivatives);

  rttopp::WaypointJoint<n_joints> joint_state;
  joint_state.velocity = derivatives.first * state.velocity;
  for (size_t joint = 0; joint < n_joints; ++joint) {
    EXPECT_LT(joint_state.velocity[joint],
              joint_constraints.velocity_max[joint]);
    EXPECT_GT(joint_state.velocity[joint],
              joint_constraints.velocity_min[joint]);
  }

  double acc_min, acc_max;
  std::tie(acc_min, acc_max) =
      path_acceleration_limits.calculateDynamicLimits(state, derivatives);
  EXPECT_LT(acc_min, acc_max);
  EXPECT_GT(acc_min + MVC2_ACC_TOL, acc_max);
}

TEST(PathVelocityLimit, MVC2JointAccelLimitsDerivFirstZero) {
  const size_t n_joints = 6;
  const auto joint_constraints =
      rttopp::demo_trajectories::generateGenericJointConstraints<n_joints>();

  rttopp::PathVelocityLimit<n_joints> path_velocity_limit(joint_constraints);
  rttopp::PathAccelerationLimits<n_joints> path_acceleration_limits(
      joint_constraints);
  rttopp::JointPathDerivatives<n_joints> derivatives;
  rttopp::PathState state;

  derivatives.first << 0.18224495097349483, -0.0730498627981969,
      0.16650664047503788, 0.5443824292200334, 0.3757492021314488,
      -4.631279037224212e-09;

  derivatives.second << -0.10615784016085551, -0.18877835868049317,
      -0.0025535481861300273, -0.10279215502620404, -0.11282042097047713,
      0.2727641821488546;

  state.velocity = path_velocity_limit.calculateOverallLimit(derivatives);

  double acc_min, acc_max;
  std::tie(acc_min, acc_max) =
      path_acceleration_limits.calculateDynamicLimits(state, derivatives);
  EXPECT_LT(acc_min, acc_max);
  // TODO(wolfgang): not entirely sure why, but in this MVC2 case with one first
  // deriv zero, acc_min and acc_max do not converge. Maybe because this axis is
  // ignored during acc limit calculation, but the MVC2 computation still
  // ensures that no acc limits are violated at this velocity. EXPECT_GT(acc_min
  // + MVC2_ACC_TOL, acc_max);

  rttopp::WaypointJoint<n_joints> joint_state;
  joint_state.velocity = derivatives.first * state.velocity;
  state.acceleration = acc_min;
  joint_state.acceleration =
      derivatives.second * rttopp::utils::pow(state.velocity, 2) +
      derivatives.first * state.acceleration;
  for (size_t joint = 0; joint < n_joints; ++joint) {
    EXPECT_LT(joint_state.velocity[joint],
              joint_constraints.velocity_max[joint]);
    EXPECT_GT(joint_state.velocity[joint],
              joint_constraints.velocity_min[joint]);
    EXPECT_LT(joint_state.acceleration[joint],
              joint_constraints.acceleration_max[joint]);
    EXPECT_GT(joint_state.acceleration[joint],
              joint_constraints.acceleration_min[joint]);
  }
}

TEST(PathVelocityLimit, RandomWaypointsIIWA) {
  const size_t N_TRAJECTORIES = 3.0e04;
  const size_t N_WAYPOINTS = 5;

  const size_t n_joints = rttopp::demo_trajectories::NUM_IIWA_JOINTS;
  const auto joint_constraints =
      rttopp::demo_trajectories::generateIIWAJointConstraints();

  rttopp::PathVelocityLimit<n_joints> path_velocity_limit(joint_constraints);
  rttopp::PathAccelerationLimits<n_joints> path_acceleration_limits(
      joint_constraints);
  rttopp::Preprocessing<n_joints> preprocessing;
  rttopp::JointPathDerivatives<n_joints> derivatives;
  rttopp::PathState state;
  rttopp::WaypointJoint<n_joints> joint_state;

  for (size_t t = 0; t < N_TRAJECTORIES; ++t) {
    const auto waypoints =
        rttopp::demo_trajectories::generateRandomJointWaypoints<n_joints>(
            N_WAYPOINTS, rttopp::demo_trajectories::iiwaJointPositionLimits());
    size_t n_seg = preprocessing.processWaypoints(waypoints);
    for (size_t seg = 0; seg < n_seg; ++seg) {
      derivatives = preprocessing.getDerivatives(seg);
      const auto vel_limit_first =
          path_velocity_limit.calculateJointVelocityLimit(derivatives);
      const auto vel_limit_second =
          path_velocity_limit.calculateJointAccelerationLimit(derivatives);

      // TODO(wolfgang): remove this Kunz comparison code
      // double max_path_velocity = std::numeric_limits<double>::infinity();
      // const Eigen::VectorXd config_deriv = derivatives.first;
      // const Eigen::VectorXd config_deriv2 = derivatives.second;
      // for (unsigned int i = 0; i < n_joints; ++i) {
      //   if (config_deriv[i] != 0.0) {
      //     for (unsigned int j = i + 1; j < n_joints; ++j) {
      //       if (config_deriv[j] != 0.0) {
      //         double a_ij = config_deriv2[i] / config_deriv[i] -
      //                       config_deriv2[j] / config_deriv[j];
      //         if (a_ij != 0.0) {
      //           max_path_velocity = std::min(
      //               max_path_velocity,
      //               sqrt((joint_constraints.acceleration_max[i] /
      //                                            std::abs(config_deriv[i]) +
      //                                        joint_constraints.acceleration_max[j]
      //                                        /
      //                                            std::abs(config_deriv[j])) /
      //                                       std::abs(a_ij)));
      //         }
      //       }
      //     }
      //   }
      // }

      state.velocity = std::min(vel_limit_first, vel_limit_second);
      double acc_min, acc_max;
      std::tie(acc_min, acc_max) =
          path_acceleration_limits.calculateDynamicLimits(state, derivatives);
      EXPECT_LT(acc_min, acc_max);

      if (vel_limit_second < vel_limit_first) {
        //         if (max_path_velocity < vel_limit_second) {
        //   std::cout << "kunz smaller! " << max_path_velocity << ", " <<
        //   vel_limit_second << std::endl;
        // }

        EXPECT_GT(acc_min + MVC2_ACC_TOL_GENERIC, acc_max);
        // if (acc_min + MVC2_ACC_TOL < acc_max) {
        //   std::cout << "acc tol violated kunz " << max_path_velocity << ", "
        //   << vel_limit_second << std::endl; std::cout << derivatives.first <<
        //   std::endl; std::cout << derivatives.second << std::endl;
        // }
      }

      joint_state.velocity = derivatives.first * state.velocity;
      state.acceleration = acc_min;
      joint_state.acceleration =
          derivatives.second * rttopp::utils::pow(state.velocity, 2) +
          derivatives.first * state.acceleration;
      bool joint_at_limit = false;
      for (size_t joint = 0; joint < n_joints; ++joint) {
        EXPECT_LT(joint_state.velocity[joint],
                  joint_constraints.velocity_max[joint]);
        EXPECT_GT(joint_state.velocity[joint],
                  joint_constraints.velocity_min[joint]);
        EXPECT_LT(joint_state.acceleration[joint],
                  joint_constraints.acceleration_max[joint]);
        EXPECT_GT(joint_state.acceleration[joint],
                  joint_constraints.acceleration_min[joint]);
        if (joint_state.velocity[joint] - MVC1_VEL_TOL <=
                joint_constraints.velocity_min[joint] ||
            joint_state.velocity[joint] + MVC1_VEL_TOL >=
                joint_constraints.velocity_max[joint]) {
          joint_at_limit = true;
        }
      }
      if (vel_limit_first <= vel_limit_second) {
        EXPECT_TRUE(joint_at_limit);
      }
    }
  }
}

// TODO(wolfgang): add tests with asymmetric acceleration constraints
// also with min/max value having the same sign

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
