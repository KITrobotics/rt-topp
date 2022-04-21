#include <gtest/gtest.h>

#include <rttopp/demo_trajectories.h>
#include <rttopp/path_acceleration_limits.h>
#include <rttopp/path_velocity_limit.h>
#include <rttopp/preprocessing.h>

constexpr double MVC1_VEL_TOL = 1.0e-02;
constexpr double MVC2_ACC_TOL = 1.0;

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

TEST(PathVelocityLimit, RandomWaypointsIIWA) {
  // TODO(wolfgang): increase these values
  const size_t N_TRAJECTORIES = 2.0e04;
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

      if (vel_limit_second < vel_limit_first) {
        //         if (max_path_velocity < vel_limit_second) {
        //   std::cout << "kunz smaller! " << max_path_velocity << ", " <<
        //   vel_limit_second << std::endl;
        // }

        double acc_min, acc_max;
        state.velocity = vel_limit_second;
        std::tie(acc_min, acc_max) =
            path_acceleration_limits.calculateDynamicLimits(state, derivatives);
        EXPECT_LT(acc_min, acc_max);
        EXPECT_GT(acc_min + MVC2_ACC_TOL, acc_max);
        // if (acc_min + MVC2_ACC_TOL < acc_max) {
        //   std::cout << "acc tol violated kunz " << max_path_velocity << ", "
        //   << vel_limit_second << std::endl; std::cout << derivatives.first <<
        //   std::endl; std::cout << derivatives.second << std::endl;
        // }
      }

      joint_state.velocity =
          derivatives.first * std::min(vel_limit_first, vel_limit_second);
      bool joint_at_limit = false;
      for (size_t joint = 0; joint < n_joints; ++joint) {
        EXPECT_LT(joint_state.velocity[joint],
                  joint_constraints.velocity_max[joint]);
        EXPECT_GT(joint_state.velocity[joint],
                  joint_constraints.velocity_min[joint]);
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
