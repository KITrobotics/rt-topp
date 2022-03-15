#include <gtest/gtest.h>

#include <rttopp/demo_trajectories.h>
#include <rttopp/path_velocity_limit.h>
#include <rttopp/preprocessing.h>

TEST(PathVelocityLimit, JointVelocityLimit) {
  const size_t n_joints = rttopp::demo_trajectories::NUM_IIWA_JOINTS;
  const auto joint_constraints =
      rttopp::demo_trajectories::generateIIWAJointConstraints();

  rttopp::PathVelocityLimit<n_joints> path_velocity_limit(joint_constraints);

  rttopp::JointPathDerivatives<n_joints> derivatives;

  double velocity_limit =
      path_velocity_limit.calculateOverallLimit(derivatives);

  std::cout << "limit " << velocity_limit;

  /*
  for (std::size_t i = 0; i < trajectory.timestamps.size(); ++i) {
    PathState path_state{};
    path_state.position = path.getPositionAsMatrix()(i);
    path_state.velocity = path.getVelocitiesAsMatrix()(i);
    path_state.acceleration = path.getAccelerationsAsMatrix()(i);
    path_state.jerk = path.getJerksAsMatrix()(i);

    JointPathDerivativeState joint_path_derivative_state;
    joint_path_derivative_state.first = derivatives.getFirstAsMatrix().col(i);
    joint_path_derivative_state.second = derivatives.getSecondAsMatrix().col(i);
    joint_path_derivative_state.third = derivatives.getThirdAsMatrix().col(i);

    const auto [_, velocity_limit] =
        path_velocity_limit.calculateJointVelocityLimit(
            path_state, joint_path_derivative_state);

    std::vector<double> joint_velocities;

    for (std::size_t j = 0; j < n_joints; ++j) {
      const auto first = derivatives.getFirstAsMatrix().coeff(j, i);
      const auto joint_velocity = velocity_limit * first;
      const auto joint_constraint = (joint_velocity < 0.0)
                                        ? joint_constraints.velocity_min(j)
                                        : joint_constraints.velocity_max(j);

      // we just hope that none of the joint constraints is violated
      if (joint_velocity > 0.0) {
        EXPECT_LT(joint_velocity - eps, joint_constraint);
      } else {
        EXPECT_GT(joint_velocity + eps, joint_constraint);
      }
      joint_velocities.push_back(joint_velocity);
    }

    const auto joint_velocity_max_it = std::max_element(
        std::begin(joint_velocities), std::end(joint_velocities));
    const auto joint_velocity_idx =
        std::distance(std::begin(joint_velocities), joint_velocity_max_it);
    const auto joint_constraint_max =
        (*joint_velocity_max_it < 0.0)
            ? joint_constraints.velocity_min(joint_velocity_idx)
            : joint_constraints.velocity_max(joint_velocity_idx);

    // we also expect that at least one of the joints is close to its max
    if (*joint_velocity_max_it > 0.0) {
      EXPECT_LT(*joint_velocity_max_it - eps, joint_constraint_max);
    } else {
      EXPECT_GT(*joint_velocity_max_it + eps, joint_constraint_max);
    }
  }
  */
}

/*
TEST(PathVelocityLimit, JointAccelerationLimitFirstZero) {
  const auto trajectory = generateInvertibleTrajectory();
  auto joint_constraints = generateIIWAJointConstraints();
  const auto joint_trajectory =
      computeJointTrajectory(trajectory, joint_constraints);
  PathProjection path(&trajectory);
  JointPathDerivatives derivatives;

  derivatives.computeDerivatives(joint_trajectory, path);

  const auto n_joints = 7;
  const auto eps = 1.0e-5;

  PathVelocityLimit path_velocity_limit(joint_constraints);

  for (std::size_t i = 0; i < trajectory.timestamps.size(); ++i) {
    PathState path_state{};
    path_state.position = path.getPositionAsMatrix()(i);
    path_state.velocity = path.getVelocitiesAsMatrix()(i);
    path_state.acceleration = path.getAccelerationsAsMatrix()(i);
    path_state.jerk = path.getJerksAsMatrix()(i);

    JointPathDerivativeState joint_path_derivative_state;
    joint_path_derivative_state.second = derivatives.getSecondAsMatrix().col(i);
    joint_path_derivative_state.third = derivatives.getThirdAsMatrix().col(i);

    // The simpler case when first derivatives are zero
    joint_path_derivative_state.first.setZero();
    std::vector<double> joint_velocities;
    const auto [_, velocity_limit] =
        path_velocity_limit.calculateJointAccelerationLimit(
            path_state, joint_path_derivative_state);

    for (std::size_t j = 0; j < n_joints; ++j) {
      if (velocity_limit == std::numeric_limits<double>::max()) continue;

      const auto second = derivatives.getSecondAsMatrix().coeff(j, i);
      const auto joint_velocity =
          second * common::utils::pow(velocity_limit, 2);
      const auto joint_constraint = (joint_velocity < 0.0)
                                        ? joint_constraints.acceleration_min(j)
                                        : joint_constraints.acceleration_max(j);

      // we just hope that none of the joint constraints is violated
      if (joint_velocity > 0.0) {
        EXPECT_LT(joint_velocity - eps, joint_constraint);
      } else {
        EXPECT_GT(joint_velocity + eps, joint_constraint);
      }
      joint_velocities.push_back(joint_velocity);
    }

    if (!joint_velocities.empty()) {
      const auto joint_velocity_max_it = std::max_element(
          std::begin(joint_velocities), std::end(joint_velocities));
      const auto joint_velocity_idx =
          std::distance(std::begin(joint_velocities), joint_velocity_max_it);
      const auto joint_constraint_max =
          (*joint_velocity_max_it < 0.0)
              ? joint_constraints.acceleration_min(joint_velocity_idx)
              : joint_constraints.acceleration_max(joint_velocity_idx);

      // we also expect that at least one of the joints is close to its max
      if (*joint_velocity_max_it > 0.0) {
        EXPECT_LT(*joint_velocity_max_it - eps, joint_constraint_max);
      } else {
        EXPECT_GT(*joint_velocity_max_it + eps, joint_constraint_max);
      }
    }
  }
}

TEST(PathVelocityLimit, JointAccelerationLimitFirstNotZero) {
  const auto trajectory = generateInvertibleTrajectory();
  auto joint_constraints = generateIIWAJointConstraints();
  const auto joint_trajectory =
      computeJointTrajectory(trajectory, joint_constraints);
  PathProjection path(&trajectory);
  JointPathDerivatives derivatives;

  derivatives.computeDerivatives(joint_trajectory, path);

  const auto n_joints = 7;
  const auto eps = 1.0e-5;

  PathVelocityLimit path_velocity_limit(joint_constraints);

  for (std::size_t i = 0; i < trajectory.timestamps.size(); ++i) {
    PathState path_state{};
    path_state.position = path.getPositionAsMatrix()(i);
    path_state.velocity = path.getVelocitiesAsMatrix()(i);
    path_state.acceleration = path.getAccelerationsAsMatrix()(i);
    path_state.jerk = path.getJerksAsMatrix()(i);

    // The full case when first and second are not zero
    // In this case a guess for the acceleration value is possible and
    // necessary
    JointPathDerivativeState joint_path_derivative_state;
    joint_path_derivative_state.first = derivatives.getFirstAsMatrix().col(i);
    joint_path_derivative_state.second = derivatives.getSecondAsMatrix().col(i);
    joint_path_derivative_state.third = derivatives.getThirdAsMatrix().col(i);

    std::vector<double> path_acceleration_first_pos_upper_guesses;
    std::vector<double> path_acceleration_first_pos_lower_guesses;
    std::vector<double> path_acceleration_first_neg_upper_guesses;
    std::vector<double> path_acceleration_first_neg_lower_guesses;

    const auto [_, velocity_limit] =
        path_velocity_limit.calculateJointAccelerationLimit(
            path_state, joint_path_derivative_state);

    for (std::size_t j = 0; j < n_joints; ++j) {
      if (velocity_limit == std::numeric_limits<double>::max()) continue;

      const auto first = derivatives.getFirstAsMatrix().coeff(j, i);
      const auto second = derivatives.getSecondAsMatrix().coeff(j, i);
      const auto path_acceleration_upper_guess =
          (joint_constraints.acceleration_max(j) -
           second * common::utils::pow(velocity_limit, 2)) /
          first;
      const auto path_acceleration_lower_guess =
          (joint_constraints.acceleration_min(j) -
           second * common::utils::pow(velocity_limit, 2)) /
          first;

      // store acceleration guesses to compare later if the upper and lower
      // bounds make sense
      if (first > 0.0) {
        EXPECT_GT(path_acceleration_upper_guess - path_acceleration_lower_guess,
                  -eps);
        path_acceleration_first_pos_upper_guesses.push_back(
            path_acceleration_upper_guess);
        path_acceleration_first_pos_lower_guesses.push_back(
            path_acceleration_lower_guess);
      } else if (first < 0.0) {
        EXPECT_GT(path_acceleration_lower_guess - path_acceleration_upper_guess,
                  -eps);
        path_acceleration_first_neg_lower_guesses.push_back(
            path_acceleration_upper_guess);
        path_acceleration_first_neg_upper_guesses.push_back(
            path_acceleration_lower_guess);
      }
    }

    if (!path_acceleration_first_pos_upper_guesses.empty()) {
      const auto min_guess = *std::min_element(
          std::begin(path_acceleration_first_pos_upper_guesses),
          std::end(path_acceleration_first_pos_upper_guesses));
      const auto max_guess = *std::max_element(
          std::begin(path_acceleration_first_pos_lower_guesses),
          std::end(path_acceleration_first_pos_lower_guesses));

      // when the first derivative is positive, we expect that the min upper
      // bound - max lower bound >= 0
      EXPECT_GT(min_guess - max_guess, -eps);
    }
    if (!path_acceleration_first_neg_upper_guesses.empty()) {
      const auto min_guess = *std::min_element(
          std::begin(path_acceleration_first_neg_upper_guesses),
          std::end(path_acceleration_first_neg_upper_guesses));
      const auto max_guess = *std::max_element(
          std::begin(path_acceleration_first_neg_lower_guesses),
          std::end(path_acceleration_first_neg_lower_guesses));

      // when the first derivative is negative, all signs are flipped and we
      // expect that the min upper bound - max lower bound <= 0
      EXPECT_GT(min_guess - max_guess, -eps);
    }
  }
}

TEST(PathVelocityLimit, JointJerkLimitFirstZeroSecondZero) {
  const auto trajectory = generateInvertibleTrajectory();
  const auto joint_constraints = generateIIWAJointConstraints();
  const auto joint_trajectory =
      computeJointTrajectory(trajectory, joint_constraints);
  PathProjection path(&trajectory);
  JointPathDerivatives derivatives;

  derivatives.computeDerivatives(joint_trajectory, path);

  const auto n_joints = 7;
  const auto eps = 1.0e-5;

  PathVelocityLimit path_velocity_limit(joint_constraints);

  for (std::size_t i = 0; i < trajectory.timestamps.size(); ++i) {
    PathState path_state;
    path_state.position = path.getPositionAsMatrix()(i);
    path_state.velocity = path.getVelocitiesAsMatrix()(i);
    path_state.acceleration = path.getAccelerationsAsMatrix()(i);
    path_state.jerk = path.getJerksAsMatrix()(i);

    JointPathDerivativeState joint_path_derivative_state;
    joint_path_derivative_state.third = derivatives.getThirdAsMatrix().col(i);

    // the simpler case when first and second derivatives are zero
    joint_path_derivative_state.second.setZero();
    joint_path_derivative_state.first.setZero();

    std::vector<double> joint_velocities;
    const auto [_, velocity_limit] =
        path_velocity_limit.calculateJointJerkLimit(
            path_state, joint_path_derivative_state);

    for (std::size_t j = 0; j < n_joints; ++j) {
      if (velocity_limit == std::numeric_limits<double>::max()) continue;

      const auto first = derivatives.getFirstAsMatrix().coeff(j, i);
      const auto second = derivatives.getSecondAsMatrix().coeff(j, i);
      const auto third = derivatives.getThirdAsMatrix().coeff(j, i);
      const auto joint_velocity = third * common::utils::pow(velocity_limit, 3);
      const auto joint_constraint = (joint_velocity < 0.0)
                                        ? joint_constraints.jerk_min(j)
                                        : joint_constraints.jerk_max(j);

      // we just hope that none of the joint constraints is violated
      if (joint_velocity > 0.0) {
        EXPECT_LT(joint_velocity - eps, joint_constraint);
      } else {
        EXPECT_GT(joint_velocity + eps, joint_constraint);
      }
      joint_velocities.push_back(joint_velocity);
    }

    if (!joint_velocities.empty()) {
      const auto joint_velocity_max_it = std::max_element(
          std::begin(joint_velocities), std::end(joint_velocities));
      const auto joint_velocity_idx =
          std::distance(std::begin(joint_velocities), joint_velocity_max_it);
      const auto joint_constraint_max =
          (*joint_velocity_max_it < 0.0)
              ? joint_constraints.jerk_min(joint_velocity_idx)
              : joint_constraints.jerk_max(joint_velocity_idx);

      // we also expect that at least one of the joints is close to its max
      if (*joint_velocity_max_it > 0.0) {
        EXPECT_LT(*joint_velocity_max_it - eps, joint_constraint_max);
      } else {
        EXPECT_GT(*joint_velocity_max_it + eps, joint_constraint_max);
      }
    }
  }
}

TEST(PathVelocityLimit, JointJerkLimitFirstZeroSecondNotZero) {
  const auto trajectory = generateInvertibleTrajectory();
  const auto joint_constraints = generateIIWAJointConstraints();
  const auto joint_trajectory =
      computeJointTrajectory(trajectory, joint_constraints);
  PathProjection path(&trajectory);
  JointPathDerivatives derivatives;

  derivatives.computeDerivatives(joint_trajectory, path);

  const auto n_joints = 7;
  const auto eps = 1.0e-5;

  PathVelocityLimit path_velocity_limit(joint_constraints);

  for (std::size_t i = 0; i < trajectory.timestamps.size(); ++i) {
    PathState path_state{};
    path_state.position = path.getPositionAsMatrix()(i);
    path_state.velocity = path.getVelocitiesAsMatrix()(i);
    path_state.acceleration = path.getAccelerationsAsMatrix()(i);
    path_state.jerk = path.getJerksAsMatrix()(i);

    JointPathDerivativeState joint_path_derivative_state;
    joint_path_derivative_state.second = derivatives.getSecondAsMatrix().col(i);
    joint_path_derivative_state.third = derivatives.getThirdAsMatrix().col(i);

    // the second case when only first derivatives are zero
    joint_path_derivative_state.first.setZero();

    std::vector<double> path_acceleration_first_pos_upper_guesses;
    std::vector<double> path_acceleration_first_pos_lower_guesses;
    std::vector<double> path_acceleration_first_neg_upper_guesses;
    std::vector<double> path_acceleration_first_neg_lower_guesses;

    const auto [_, velocity_limit] =
        path_velocity_limit.calculateJointJerkLimit(
            path_state, joint_path_derivative_state);

    for (std::size_t j = 0; j < n_joints; ++j) {
      if (velocity_limit == std::numeric_limits<double>::max()) continue;

      const auto first = derivatives.getFirstAsMatrix().coeff(j, i);
      const auto second = derivatives.getSecondAsMatrix().coeff(j, i);
      const auto third = derivatives.getThirdAsMatrix().coeff(j, i);

      const auto path_acceleration_upper_guess =
          (joint_constraints.jerk_max(j) -
           third * common::utils::pow(velocity_limit, 3)) /
          (3.0 * second * velocity_limit);
      const auto path_acceleration_lower_guess =
          (joint_constraints.jerk_min(j) -
           third * common::utils::pow(velocity_limit, 3)) /
          (3.0 * second * velocity_limit);

      // store acceleration guesses to compare later if the upper and lower
      // bounds make sense
      if (second > 0.0) {
        EXPECT_GT(path_acceleration_upper_guess - path_acceleration_lower_guess,
                  -eps);
        path_acceleration_first_pos_upper_guesses.push_back(
            path_acceleration_upper_guess);
        path_acceleration_first_pos_lower_guesses.push_back(
            path_acceleration_lower_guess);
      } else if (second < 0.0) {
        EXPECT_GT(path_acceleration_lower_guess - path_acceleration_upper_guess,
                  -eps);
        path_acceleration_first_neg_lower_guesses.push_back(
            path_acceleration_upper_guess);
        path_acceleration_first_neg_upper_guesses.push_back(
            path_acceleration_lower_guess);
      }
    }

    if (!path_acceleration_first_pos_upper_guesses.empty()) {
      const auto min_guess = *std::min_element(
          std::begin(path_acceleration_first_pos_upper_guesses),
          std::end(path_acceleration_first_pos_upper_guesses));
      const auto max_guess = *std::max_element(
          std::begin(path_acceleration_first_pos_lower_guesses),
          std::end(path_acceleration_first_pos_lower_guesses));

      // when the first derivative is positive, we expect that the min upper
      // bound - max lower bound >= 0
      EXPECT_GT(min_guess - max_guess, -eps);
    }
    if (!path_acceleration_first_neg_upper_guesses.empty()) {
      const auto min_guess = *std::min_element(
          std::begin(path_acceleration_first_neg_upper_guesses),
          std::end(path_acceleration_first_neg_upper_guesses));
      const auto max_guess = *std::max_element(
          std::begin(path_acceleration_first_neg_lower_guesses),
          std::end(path_acceleration_first_neg_lower_guesses));

      // when the first derivative is negative, all signs are flipped and we
      // expect that the min upper bound - max lower bound <= 0
      EXPECT_GT(min_guess - max_guess, -eps);
    }
  }
}

TEST(PathVelocityLimit, JointJerkLimitFirstNotZeroSecondNotZero) {
  const auto trajectory = generateInvertibleTrajectory();
  const auto joint_constraints = generateIIWAJointConstraints();
  const auto joint_trajectory =
      computeJointTrajectory(trajectory, joint_constraints);
  PathProjection path(&trajectory);
  JointPathDerivatives derivatives;

  derivatives.computeDerivatives(joint_trajectory, path);

  const auto n_joints = 7;
  const auto eps = 1.0e-5;

  PathVelocityLimit path_velocity_limit(joint_constraints);

  for (std::size_t t = 0; t < trajectory.timestamps.size(); ++t) {
    PathState path_state{};
    path_state.position = path.getPositionAsMatrix()(t);
    path_state.velocity = path.getVelocitiesAsMatrix()(t);
    path_state.acceleration = path.getAccelerationsAsMatrix()(t);
    path_state.jerk = path.getJerksAsMatrix()(t);

    JointPathDerivativeState joint_path_derivative_state;
    joint_path_derivative_state.first = derivatives.getFirstAsMatrix().col(t);
    joint_path_derivative_state.second = derivatives.getSecondAsMatrix().col(t);
    joint_path_derivative_state.third = derivatives.getThirdAsMatrix().col(t);

    std::vector<double> path_acceleration_first_pos_upper_guesses;
    std::vector<double> path_acceleration_first_pos_lower_guesses;
    std::vector<double> path_acceleration_first_neg_upper_guesses;
    std::vector<double> path_acceleration_first_neg_lower_guesses;

    std::vector<double> path_jerk_first_pos_upper_guesses;
    std::vector<double> path_jerk_first_pos_lower_guesses;
    std::vector<double> path_jerk_first_neg_upper_guesses;
    std::vector<double> path_jerk_first_neg_lower_guesses;

    const auto [_, velocity_limit] =
        path_velocity_limit.calculateJointJerkLimit(
            path_state, joint_path_derivative_state);

    for (std::size_t i = 0; i < n_joints; ++i) {
      if (velocity_limit == std::numeric_limits<double>::max()) continue;

      const auto first_i = derivatives.getFirstAsMatrix().coeff(i, t);
      const auto second_i = derivatives.getSecondAsMatrix().coeff(i, t);
      const auto third_i = derivatives.getThirdAsMatrix().coeff(i, t);

      for (std::size_t j = 0; j < n_joints; ++j) {
        if (i == j) continue;

        const auto first_j = derivatives.getFirstAsMatrix().coeff(j, t);
        const auto second_j = derivatives.getSecondAsMatrix().coeff(j, t);
        const auto third_j = derivatives.getThirdAsMatrix().coeff(j, t);

        const auto first_term_num =
            (joint_constraints.jerk_min[j] -
             third_j * common::utils::pow(velocity_limit, 3)) /
            first_j;

        const auto second_term_num =
            (joint_constraints.jerk_max[i] -
             third_i * common::utils::pow(velocity_limit, 3)) /
            first_i;

        const auto num = (first_term_num - second_term_num);
        const auto den = ((3.0 * second_j * velocity_limit) / first_j) -
                         ((3.0 * second_i * velocity_limit) / first_i);

        const auto path_acceleration_guess = num / den;

        if (first_i > 0.0 && first_j > 0.0) {
          if (den < 0.0) {
            path_acceleration_first_pos_upper_guesses.push_back(
                path_acceleration_guess);
          } else {
            path_acceleration_first_pos_lower_guesses.push_back(
                path_acceleration_guess);
          }
        } else if (first_i < 0.0 && first_j < 0.0) {
          if (den > 0.0) {
            path_acceleration_first_neg_upper_guesses.push_back(
                path_acceleration_guess);
          } else {
            path_acceleration_first_neg_lower_guesses.push_back(
                path_acceleration_guess);
          }
        }

        const auto path_jerk_upper_guess_i =
            (joint_constraints.jerk_max[i] -
             third_i * common::utils::pow(velocity_limit, 3) -
             3.0 * second_i * velocity_limit * path_acceleration_guess) /
            first_i;

        const auto path_jerk_lower_guess_i =
            (joint_constraints.jerk_min[i] -
             third_i * common::utils::pow(velocity_limit, 3) -
             3.0 * second_i * velocity_limit * path_acceleration_guess) /
            first_i;

        const auto path_jerk_upper_guess_j =
            (joint_constraints.jerk_max[j] -
             third_j * common::utils::pow(velocity_limit, 3) -
             3.0 * second_j * velocity_limit * path_acceleration_guess) /
            first_j;

        const auto path_jerk_lower_guess_j =
            (joint_constraints.jerk_min[j] -
             third_j * common::utils::pow(velocity_limit, 3) -
             3.0 * second_j * velocity_limit * path_acceleration_guess) /
            first_j;

        if (first_i > 0.0) {
          EXPECT_GT(path_jerk_upper_guess_i - path_jerk_lower_guess_i, -eps);
        } else {
          EXPECT_GT(path_jerk_lower_guess_i - path_jerk_upper_guess_i, -eps);
        }

        if (first_j > 0.0) {
          EXPECT_GT(path_jerk_upper_guess_j - path_jerk_lower_guess_j, -eps);
        } else {
          EXPECT_GT(path_jerk_lower_guess_j - path_jerk_upper_guess_j, -eps);
        }
      }

      if (!path_acceleration_first_pos_upper_guesses.empty() &&
          !path_acceleration_first_pos_lower_guesses.empty()) {
        const auto min_upper_guess = *std::min_element(
            std::begin(path_acceleration_first_pos_upper_guesses),
            std::end(path_acceleration_first_pos_upper_guesses));
        const auto max_lower_guess = *std::max_element(
            std::begin(path_acceleration_first_pos_lower_guesses),
            std::end(path_acceleration_first_pos_lower_guesses));

        // when the first derivative is positive, we expect that the min upper
        // bound - max lower bound >= 0
        EXPECT_GT(min_upper_guess - max_lower_guess, -eps);
      }

      if (!path_acceleration_first_neg_upper_guesses.empty() &&
          !path_acceleration_first_neg_lower_guesses.empty()) {
        const auto min_upper_guess = *std::min_element(
            std::begin(path_acceleration_first_neg_upper_guesses),
            std::end(path_acceleration_first_neg_upper_guesses));
        const auto max_lower_guess = *std::max_element(
            std::begin(path_acceleration_first_neg_lower_guesses),
            std::end(path_acceleration_first_neg_lower_guesses));

        // when the first derivative is positive, we expect that the min upper
        // bound - max lower bound >= 0
        EXPECT_GT(min_upper_guess - max_lower_guess, -eps);
      }
    }
  }
}
*/

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
