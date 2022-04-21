#pragma once

#include <unsupported/Eigen/Polynomials>

#include <rttopp/types_utils.h>

namespace rttopp {

template <size_t N_JOINTS>
class PathVelocityLimit {
  using JointPathDerivativeState = JointPathDerivatives<N_JOINTS>;

 public:
  explicit PathVelocityLimit(
      const JointConstraints<N_JOINTS> &joint_constraints)
      : joint_constraints_(joint_constraints) {}

  /**
   * @brief Calculates overall path velocity limits when considering joint
   * velocity and acceleration constraints. Internally calls
   * calculateJointVelocityLimit and calculateJointAccelerationLimit.
   *
   * @param path_state
   * @param joint_path_derivative_state
   *
   * @return Maximum overall velocity limit
   */
  [[nodiscard]] double calculateOverallLimit(
      const JointPathDerivativeState &joint_path_derivative_state);

  /**
   * @brief Calculates path velocity limits based only on joint velocity
   * constraints
   *
   * @param path_state
   * @param joint_path_derivative_state
   *
   * @return Maximum path velocity limit when considering only joint velocity
   * constraints
   */
  [[nodiscard]] double calculateJointVelocityLimit(
      const JointPathDerivativeState &joint_path_derivative_state);

  /**
   * @brief Calculates path velocity limits based only on joint acceleration
   * constraints
   *
   * @param path_state
   * @param joint_path_derivative_state
   *
   * @return Maximum path velocity limit when considering only joint
   * acceleration constraints
   */
  [[nodiscard]] double calculateJointAccelerationLimit(
      const JointPathDerivativeState &joint_path_derivative_state);

  /**
   * @brief Calculates path velocity limits based only on joint jerk
   * constraints
   *
   * @param path_state
   * @param joint_path_derivative_state
   *
   * @return Minimum and maximum path velocity limit respectively when
   * considering only joint jerk constraints
   */
  std::pair<double, double> calculateJointJerkLimit(
      const PathState &path_state,
      const JointPathDerivativeState &joint_path_derivative_state);

 private:
  /**
   * @brief Calculates path velocity limits based only on joint acceleration
   * constraints. This method handles the case for which both qs and qss are not
   * equal to zero
   *
   * @param path_state
   * @param joint_path_derivative_state
   *
   * @return Maximum path velocity limit
   */
  double calculateJointAccelerationLimit1(
      const JointPathDerivativeState &joint_path_derivative_state);

  /**
   * @brief Calculates path velocity limits based only on joint acceleration
   * constraints. This method handles the case for which qs is equal to zero and
   * qss is not equal to zero
   *
   * @param path_state
   * @param joint_path_derivative_state
   *
   * @return Maximum path velocity limit
   */
  double calculateJointAccelerationLimit2(
      const JointPathDerivativeState &joint_path_derivative_state);

  /**
   * @brief Calculates path velocity limits based only on joint jerk
   * constraints. This method handles the case for which both qs and qss are
   * equal to zero and qsss is not equal to zero
   *
   * @param path_state
   * @param joint_path_derivative_state
   *
   * @return Minimum and maximum path velocity limit respectively
   */
  std::pair<double, double> calculateJointJerkLimit1(
      const PathState &path_state,
      const JointPathDerivativeState &joint_path_derivative_state);

  /**
   * @brief Calculates path velocity limits based only on joint jerk
   * constraints. This method handles the case for which both qss and qsss are
   * not equal to zero and qs is equal to zero
   *
   * @param path_state
   * @param joint_path_derivative_state
   *
   * @return Minimum and maximum path velocity limit respectively
   */
  std::pair<double, double> calculateJointJerkLimit2(
      const PathState &path_state,
      const JointPathDerivativeState &joint_path_derivative_state);

  /**
   * @brief Calculates path velocity limits based only on joint jerk
   * constraints. This method handles the case for which qs, qss and qsss are
   * not equal to zero
   *
   * @param path_state
   * @param joint_path_derivative_state
   *
   * @return Minimum and maximum path velocity limit respectively
   */
  std::pair<double, double> calculateJointJerkLimit3(
      const PathState &path_state,
      const JointPathDerivativeState &joint_path_derivative_state);

  std::pair<double, double> calculateJointJerkLimit3_1(
      const PathState &path_state,
      const JointPathDerivativeState &joint_path_derivative_state);

  std::pair<double, double> calculateJointJerkLimit3_2(
      const PathState &path_state,
      const JointPathDerivativeState &joint_path_derivative_state);

  std::pair<double, double> calculateJointJerkLimit3_3(
      const PathState &path_state,
      const JointPathDerivativeState &joint_path_derivative_state);

  const JointConstraints<N_JOINTS> &joint_constraints_;
};

template <size_t N_JOINTS>
double PathVelocityLimit<N_JOINTS>::calculateOverallLimit(
    const JointPathDerivativeState &joint_path_derivative_state) {
  const auto upper_velocity =
      calculateJointVelocityLimit(joint_path_derivative_state);
  const auto upper_acceleration =
      calculateJointAccelerationLimit(joint_path_derivative_state);

  const auto overall_limit = std::min(upper_velocity, upper_acceleration);

  assert(overall_limit > 0.0 &&
         overall_limit < std::numeric_limits<double>::max());

  return overall_limit;
}

template <size_t N_JOINTS>
double PathVelocityLimit<N_JOINTS>::calculateJointVelocityLimit(
    const JointPathDerivativeState &joint_path_derivative_state) {
  double velocity_max = std::numeric_limits<double>::max();

  for (size_t i = 0; i < N_JOINTS; ++i) {
    const auto first_i = joint_path_derivative_state.first(i);

    // just modify values if they are not zero, otherwise the joint constraints
    // are fulfilled anyway
    if (utils::isZero(first_i)) {
      continue;
    }

    const auto velocity_abs_max = (first_i > 0.0)
                                      ? joint_constraints_.velocity_max[i]
                                      : joint_constraints_.velocity_min[i];

    velocity_max = std::min(velocity_max, velocity_abs_max / first_i);
  }

  return velocity_max - utils::EPS;
}

template <size_t N_JOINTS>
double PathVelocityLimit<N_JOINTS>::calculateJointAccelerationLimit(
    const JointPathDerivativeState &joint_path_derivative_state) {
  const auto upper1 =
      calculateJointAccelerationLimit1(joint_path_derivative_state);
  const auto upper2 =
      calculateJointAccelerationLimit2(joint_path_derivative_state);

  return std::min(upper1, upper2) - utils::EPS;
}

template <size_t N_JOINTS>
double PathVelocityLimit<N_JOINTS>::calculateJointAccelerationLimit1(
    const JointPathDerivativeState &joint_path_derivative_state) {
  double velocity_max = std::numeric_limits<double>::max();

  for (size_t i = 0; i < N_JOINTS; ++i) {
    const auto first_i = joint_path_derivative_state.first(i);
    const auto second_i = joint_path_derivative_state.second(i);
    const auto cond_i = !utils::isZero(first_i);

    if (!cond_i) {
      continue;
    }

    const auto acceleration_abs_max_i =
        (first_i > 0.0) ? joint_constraints_.acceleration_max[i]
                        : joint_constraints_.acceleration_min[i];

    for (size_t j = i + 1; j < N_JOINTS; ++j) {
      const auto first_j = joint_path_derivative_state.first(j);
      const auto second_j = joint_path_derivative_state.second(j);
      const auto cond_j = !utils::isZero(first_j);

      if (!cond_j) {
        continue;
      }

      const auto acceleration_abs_min_j =
          (first_j > 0.0) ? joint_constraints_.acceleration_min[j]
                          : joint_constraints_.acceleration_max[j];

      const auto a = std::abs((second_i / first_i) - (second_j / first_j));
      if (utils::isZero(a)) {
        continue;
      }
      const auto b = (acceleration_abs_max_i / first_i) -
                     (acceleration_abs_min_j / first_j);

      // b > 0.0 check should also cover cases where acc_min > 0 or acc_max < 0
      if (b > 0.0) {
        const auto velocity_candidate = std::sqrt(b / a);
        velocity_max = std::min(velocity_max, velocity_candidate);
      }
    }
  }

  return velocity_max;
}

template <size_t N_JOINTS>
double PathVelocityLimit<N_JOINTS>::calculateJointAccelerationLimit2(
    const JointPathDerivativeState &joint_path_derivative_state) {
  double velocity_max = std::numeric_limits<double>::max();

  for (std::size_t i = 0; i < N_JOINTS; ++i) {
    const auto first_i = joint_path_derivative_state.first(i);
    const auto second_i = joint_path_derivative_state.second(i);
    const auto cond_i = utils::isZero(first_i) && !utils::isZero(second_i);

    if (!cond_i) {
      continue;
    }

    if (second_i < 0.0 && joint_constraints_.acceleration_min[i] < 0.0) {
      std::min(velocity_max,
               std::sqrt(joint_constraints_.acceleration_min[i] / second_i));
    } else if (second_i > 0.0 && joint_constraints_.acceleration_max[i] > 0.0) {
      std::min(velocity_max,
               std::sqrt(joint_constraints_.acceleration_max[i] / second_i));
    }
  }

  return velocity_max;
}

// TODO(wolfgang): check all formulas related to jerk and remove min velocity
// parts
template <size_t N_JOINTS>
std::pair<double, double> PathVelocityLimit<N_JOINTS>::calculateJointJerkLimit(
    const PathState &path_state,
    const JointPathDerivativeState &joint_path_derivative_state) {
  const auto [lower1, upper1] =
      calculateJointJerkLimit1(path_state, joint_path_derivative_state);
  const auto [lower2, upper2] =
      calculateJointJerkLimit2(path_state, joint_path_derivative_state);
  const auto [lower3, upper3] =
      calculateJointJerkLimit3(path_state, joint_path_derivative_state);

  return std::make_pair(std::max({lower1, lower2, lower3}),
                        std::min({upper1, upper2, upper3}));
}

template <size_t N_JOINTS>
std::pair<double, double> PathVelocityLimit<N_JOINTS>::calculateJointJerkLimit1(
    const PathState & /*path_state*/,
    const JointPathDerivativeState &joint_path_derivative_state) {
  const auto n_joints = joint_path_derivative_state.first.rows();
  double velocity_min = std::numeric_limits<double>::lowest();
  double velocity_max = std::numeric_limits<double>::max();

  for (std::size_t i = 0; i < n_joints; ++i) {
    const auto first_i = joint_path_derivative_state.first(i);
    const auto second_i = joint_path_derivative_state.second(i);
    const auto third_i = joint_path_derivative_state.third(i);

    const auto cond_i = !utils::isZero(third_i) && utils::isZero(second_i) &&
                        utils::isZero(first_i);

    if (!cond_i) {
      continue;
    }

    velocity_min =
        (third_i > 0.0)
            ? std::max(velocity_max,
                       std::cbrt(joint_constraints_.jerk_min(i) / third_i))
            : std::max(velocity_max,
                       std::cbrt(joint_constraints_.jerk_max(i) / third_i));

    velocity_max =
        (third_i > 0.0)
            ? std::min(velocity_max,
                       std::cbrt(joint_constraints_.jerk_max(i) / third_i))
            : std::min(velocity_max,
                       std::cbrt(joint_constraints_.jerk_min(i) / third_i));
  }

  return std::pair(std::max(0.0, velocity_min), velocity_max);
}

template <size_t N_JOINTS>
std::pair<double, double> PathVelocityLimit<N_JOINTS>::calculateJointJerkLimit2(
    const PathState & /*path_state*/,
    const JointPathDerivativeState &joint_path_derivative_state) {
  const auto n_joints = joint_path_derivative_state.first.rows();
  double velocity_min = std::numeric_limits<double>::lowest();
  double velocity_max = std::numeric_limits<double>::max();

  for (std::size_t i = 0; i < n_joints; ++i) {
    const auto first_i = joint_path_derivative_state.first(i);
    const auto second_i = joint_path_derivative_state.second(i);
    const auto third_i = joint_path_derivative_state.third(i);
    const auto cond_i = !utils::isZero(third_i) && !utils::isZero(second_i) &&
                        utils::isZero(first_i);

    if (!cond_i) {
      continue;
    }

    const auto jerk_abs_max_i = (second_i > 0) ? joint_constraints_.jerk_max[i]
                                               : joint_constraints_.jerk_min[i];

    for (std::size_t j = 0; j < n_joints; ++j) {
      if (i == j) {
        continue;
      }

      const auto first_j = joint_path_derivative_state.first(j);
      const auto second_j = joint_path_derivative_state.second(j);
      const auto third_j = joint_path_derivative_state.third(j);
      const auto cond_j = !utils::isZero(third_j) && !utils::isZero(second_j) &&
                          utils::isZero(first_j);

      if (!cond_j) {
        continue;
      }

      const auto jerk_abs_min_j = (second_j > 0)
                                      ? joint_constraints_.jerk_min[j]
                                      : joint_constraints_.jerk_max[j];
      const auto a =
          (third_j / (3.0 * second_j)) - (third_i / (3.0 * second_i));
      const auto b = (jerk_abs_max_i / (3.0 * second_i)) -
                     (jerk_abs_min_j / (3.0 * second_j));

      // we can only compare upper bounds to lower bounds and vice-versa
      // if (std::signbit(second_i) != std::signbit(second_j))
      //  continue;

      if (utils::isZero(a)) {
        continue;
      }

      const auto velocity_candidate = std::cbrt(-b / a);

      if (a < 0.0 && b > 0.0) {
        velocity_max = std::min(velocity_max, velocity_candidate);
      } else if (a > 0.0 && b < 0.0) {
        velocity_min = std::max(velocity_min, velocity_candidate);
      }
    }
  }

  return std::make_pair(std::max(0.0, velocity_min), velocity_max);
}

template <size_t N_JOINTS>
std::pair<double, double> PathVelocityLimit<N_JOINTS>::calculateJointJerkLimit3(
    const PathState &path_state,
    const JointPathDerivativeState &joint_path_derivative_state) {
  // return calculateJointJerkLimit3_1(path_state, joint_path_derivative_state);
  // return calculateJointJerkLimit3_2(path_state, joint_path_derivative_state);
  return calculateJointJerkLimit3_3(path_state, joint_path_derivative_state);
}

template <size_t N_JOINTS>
std::pair<double, double>
PathVelocityLimit<N_JOINTS>::calculateJointJerkLimit3_1(
    const PathState & /*path_state*/,
    const JointPathDerivativeState &joint_path_derivative_state) {
  const auto n_joints = joint_path_derivative_state.first.rows();
  double velocity_min = std::numeric_limits<double>::lowest();
  double velocity_max = std::numeric_limits<double>::max();

  for (std::size_t i = 0; i < n_joints; ++i) {
    const auto first_i = joint_path_derivative_state.first(i);
    const auto second_i = joint_path_derivative_state.second(i);
    const auto third_i = joint_path_derivative_state.third(i);
    const auto cond_i = !utils::isZero(third_i) && !utils::isZero(second_i) &&
                        !utils::isZero(first_i);

    if (!cond_i) {
      continue;
    }

    const auto jerk_abs_max_i = (first_i > 0.0)
                                    ? joint_constraints_.jerk_max(i)
                                    : joint_constraints_.jerk_min(i);

    for (std::size_t j = 0; j < n_joints; ++j) {
      if (i == j) {
        continue;
      }

      const auto first_j = joint_path_derivative_state.first(j);
      const auto second_j = joint_path_derivative_state.second(j);
      const auto third_j = joint_path_derivative_state.third(j);
      const auto cond_j = !utils::isZero(third_j) && !utils::isZero(second_j) &&
                          !utils::isZero(first_j);

      if (!cond_j) {
        continue;
      }

      const auto jerk_abs_min_j = (first_j > 0.0)
                                      ? joint_constraints_.jerk_min(j)
                                      : joint_constraints_.jerk_max(j);

      for (std::size_t k = 0; k < n_joints; ++k) {
        // if (i == k || j == k)
        //  continue;

        const auto first_k = joint_path_derivative_state.first(k);
        const auto second_k = joint_path_derivative_state.second(k);
        const auto third_k = joint_path_derivative_state.third(k);
        const auto cond_k = !utils::isZero(third_k) &&
                            !utils::isZero(second_k) && !utils::isZero(first_k);

        if (!cond_k) {
          continue;
        }

        const auto acceleration_abs_min_k =
            (first_k > 0.0) ? joint_constraints_.acceleration_min(k)
                            : joint_constraints_.acceleration_max(k);
        const auto acceleration_abs_max_k =
            (first_k > 0.0) ? joint_constraints_.acceleration_max(k)
                            : joint_constraints_.acceleration_min(k);

        const auto first_term_num_a1 =
            (third_j / first_j) - (third_i / first_i);
        const auto first_term_num_a2 = -first_term_num_a1;
        const auto first_term_den_a =
            ((3.0 * second_j / first_j) - (3.0 * second_i / first_i));
        const auto is_upper_bound = first_term_den_a > 0;
        // const auto is_upper_bound = true;
        const auto first_term_num_a =
            is_upper_bound ? first_term_num_a1 : first_term_num_a2;

        const auto first_term_a = first_term_num_a / first_term_den_a;
        const auto second_term_a = second_k / first_k;
        const auto a1 = first_term_a - second_term_a;
        const auto a2 = first_term_a + second_term_a;
        const auto a = is_upper_bound ? a1 : a2;

        const auto b1 = acceleration_abs_max_k / first_k;
        const auto b2 = -acceleration_abs_min_k / first_k;
        const auto b = is_upper_bound ? b1 : b2;

        const auto num_c1 =
            (jerk_abs_max_i / first_i) - (jerk_abs_min_j / first_j);
        const auto num_c2 = -num_c1;
        const auto num_c = is_upper_bound ? num_c1 : num_c2;
        const auto den_c = first_term_den_a;
        const auto c = num_c / den_c;

        const auto bound_a =
            -4.0 * utils::pow(b, 3) / (27.0 * utils::pow(c, 2));

        Eigen::PolynomialSolver<double, Eigen::Dynamic> solver;
        Eigen::Vector4d coeff(c, b, 0.0, a);
        std::vector<double> real_roots(3, 0.0);
        solver.compute(coeff);
        solver.realRoots(real_roots);
        std::sort(real_roots.begin(), real_roots.end());

        if (c < 0) {
          if (b > 0) {
            if (bound_a < a && a < 0) {
              velocity_max = std::min(velocity_max, real_roots.at(2));
            }
          }
        } else {  // c > 0
          if (b < 0) {
            if (a < 0) {
              velocity_max = std::min(velocity_max, real_roots.at(0));
            } else if (0 < a && a < bound_a) {
              velocity_max = std::min(velocity_max, real_roots.at(1));
            }
          } else {  // b > 0
            if (a < bound_a) {
              velocity_max = std::min(velocity_max, real_roots.at(0));
            } else if (bound_a <= a && a < 0) {
              velocity_max = std::min(velocity_max, real_roots.at(2));
            }
          }
        }
      }
    }
  }

  return std::make_pair(std::max(0.0, velocity_min), velocity_max);
}

template <size_t N_JOINTS>
std::pair<double, double>
PathVelocityLimit<N_JOINTS>::calculateJointJerkLimit3_2(
    const PathState & /*path_state*/,
    const JointPathDerivativeState &joint_path_derivative_state) {
  const auto n_joints = joint_path_derivative_state.first.rows();
  double velocity_min = std::numeric_limits<double>::lowest();
  double velocity_max = std::numeric_limits<double>::max();

  for (std::size_t i = 0; i < n_joints; ++i) {
    const auto first_i = joint_path_derivative_state.first(i);
    const auto second_i = joint_path_derivative_state.second(i);
    const auto third_i = joint_path_derivative_state.third(i);
    const auto cond_i = !utils::isZero(third_i) && !utils::isZero(second_i) &&
                        !utils::isZero(first_i);

    if (!cond_i) {
      continue;
    }

    const auto jerk_abs_max_i = (first_i > 0.0)
                                    ? joint_constraints_.jerk_max(i)
                                    : joint_constraints_.jerk_min(i);

    for (std::size_t j = 0; j < n_joints; ++j) {
      if (i == j) {
        continue;
      }

      const auto first_j = joint_path_derivative_state.first(j);
      const auto second_j = joint_path_derivative_state.second(j);
      const auto third_j = joint_path_derivative_state.third(j);
      const auto cond_j = !utils::isZero(third_j) && !utils::isZero(second_j) &&
                          !utils::isZero(first_j);

      if (!cond_j) {
        continue;
      }

      const auto jerk_abs_min_j = (first_j > 0.0)
                                      ? joint_constraints_.jerk_min(j)
                                      : joint_constraints_.jerk_max(j);

      // we can only compare upper bounds to lower bounds and vice-versa
      // within each pair this must also be valid
      // if (std::signbit(first_i) != std::signbit(first_j))
      //  continue;

      // we assume the left part of the path velocity inequation must
      // always be an upper bound to have a valid comparison, if this is not
      // the case in the path acceleration inequation no need to proceed
      const auto acc_a1 =
          ((3.0 * second_j / first_j) - (3.0 * second_i / first_i));
      const auto first_pair_is_upper = acc_a1 < 0.0;
      const auto first_pair_is_lower = !first_pair_is_upper;

      for (std::size_t k = 0; k < n_joints; ++k) {
        const auto first_k = joint_path_derivative_state.first(k);
        const auto second_k = joint_path_derivative_state.second(k);
        const auto third_k = joint_path_derivative_state.third(k);
        const auto cond_k = !utils::isZero(third_k) &&
                            !utils::isZero(second_k) && !utils::isZero(first_k);

        if (!cond_k) {
          continue;
        }

        const auto jerk_abs_max_k = (first_k > 0.0)
                                        ? joint_constraints_.jerk_max(k)
                                        : joint_constraints_.jerk_min(k);

        for (std::size_t h = 0; h < n_joints; ++h) {
          if (k == h) {
            continue;
          }

          const auto first_h = joint_path_derivative_state.first(h);
          const auto second_h = joint_path_derivative_state.second(h);
          const auto third_h = joint_path_derivative_state.third(h);
          const auto cond_h = !utils::isZero(third_h) &&
                              !utils::isZero(second_h) &&
                              !utils::isZero(first_h);

          if (!cond_h) {
            continue;
          }

          const auto jerk_abs_min_h = (first_h > 0.0)
                                          ? joint_constraints_.jerk_min(h)
                                          : joint_constraints_.jerk_max(h);
          // we can only compare upper bounds to lower bounds and vice-versa
          // within each pair this must also be valid
          // if (std::signbit(first_k) != std::signbit(first_h))
          //  continue;

          // we assume the right part of the path velocity inequation must
          // always be a lower bound to have a valid comparison, if this is
          // not the case in the path acceleration inequation no need to proceed
          const auto acc_a2 =
              ((3.0 * second_h / first_h) - (3.0 * second_k / first_k));
          const auto second_pair_is_upper = acc_a2 < 0.0;
          const auto second_pair_is_lower = !second_pair_is_upper;

          if (first_pair_is_lower || second_pair_is_upper) {
            continue;
          }
          // if (std::signbit(acc_a1) != std::signbit(acc_a2))
          //  continue;
          // if (first_pair_is_lower || second_pair_is_lower)
          //  continue;

          const auto first_term_num_a =
              (third_i / first_i) - (third_j / first_j);
          const auto first_term_den_a =
              (3.0 * second_j / first_j) - (3.0 * second_i / first_i);
          const auto second_term_num_a =
              (third_h / first_h) - (third_k / first_k);
          const auto second_term_den_a =
              (3.0 * second_h / first_h) - (3.0 * second_k / first_k);
          const auto a = (first_term_num_a / first_term_den_a) +
                         (second_term_num_a / second_term_den_a);

          const auto first_term_num_b =
              (jerk_abs_min_j / first_j) - (jerk_abs_max_i / first_i);
          const auto first_term_den_b = first_term_den_a;
          const auto second_term_num_b =
              (jerk_abs_max_k / first_k) - (jerk_abs_min_h / first_h);
          const auto second_term_den_b = second_term_den_a;
          const auto b = (first_term_num_b / first_term_den_b) +
                         (second_term_num_b / second_term_den_b);

          if (utils::isZero(first_term_den_a) ||
              utils::isZero(second_term_den_a) ||
              utils::isZero(first_term_den_b) ||
              utils::isZero(second_term_den_b)) {
            continue;
          }

          if (utils::isZero(a) || utils::isZero(b)) {
            continue;
          }

          const auto velocity_candidate = std::cbrt(-b / a);

          if (a < 0.0 && b > 0.0) {
            velocity_max = std::min(velocity_max, velocity_candidate);
          } else if (a > 0.0 && b < 0.0) {
            velocity_min = std::max(velocity_min, velocity_candidate);
          }

          // TODO(wolfgang): remove this continue
          // ??
          continue;

          if (velocity_candidate > 0.0) {
            // for consistency reasons the found path velocity is plugged back
            // to check if the upper acceleration bound >= lower acceleration
            // bound
            const auto first_term_num =
                (((jerk_abs_min_j -
                   third_j * utils::pow(velocity_candidate, 3)) /
                  first_j) -
                 ((jerk_abs_max_i -
                   third_i * utils::pow(velocity_candidate, 3)) /
                  first_i));
            const auto first_term_den =
                ((3 * second_j * velocity_candidate / first_j) -
                 (3 * second_i * velocity_candidate / first_i));
            const auto first_term = first_term_num / first_term_den;

            const auto second_term_num =
                (((jerk_abs_min_h -
                   third_h * utils::pow(velocity_candidate, 3)) /
                  first_h) -
                 ((jerk_abs_max_k -
                   third_k * utils::pow(velocity_candidate, 3)) /
                  first_k));
            const auto second_term_den =
                ((3 * second_h * velocity_candidate / first_h) -
                 (3 * second_k * velocity_candidate / first_k));
            const auto second_term = second_term_num / second_term_den;

            const auto diff = first_term - second_term;
            const auto cond_upper = first_pair_is_upper && (diff > -utils::EPS);
            const auto cond_lower = first_pair_is_lower && (diff < utils::EPS);

            if (cond_upper || cond_lower) {
              velocity_max = std::min(velocity_max, velocity_candidate);
            }
          }
        }
      }
    }
  }

  return std::make_pair(std::max(0.0, velocity_min), velocity_max);
}

template <size_t N_JOINTS>
std::pair<double, double>
PathVelocityLimit<N_JOINTS>::calculateJointJerkLimit3_3(
    const PathState &path_state,
    const JointPathDerivativeState &joint_path_derivative_state) {
  const auto n_joints = joint_path_derivative_state.first.rows();
  double velocity_min = std::numeric_limits<double>::lowest();
  double velocity_max = std::numeric_limits<double>::max();

  for (std::size_t i = 0; i < n_joints; ++i) {
    const auto first_i = joint_path_derivative_state.first(i);
    const auto second_i = joint_path_derivative_state.second(i);
    const auto third_i = joint_path_derivative_state.third(i);
    const auto cond_i = !utils::isZero(third_i) && !utils::isZero(second_i) &&
                        !utils::isZero(first_i);

    if (!cond_i) {
      continue;
    }

    const auto jerk_abs_max_i = (first_i > 0.0)
                                    ? joint_constraints_.jerk_max(i)
                                    : joint_constraints_.jerk_min(i);

    for (std::size_t j = 0; j < n_joints; ++j) {
      if (i == j) {
        continue;
      }

      const auto first_j = joint_path_derivative_state.first(j);
      const auto second_j = joint_path_derivative_state.second(j);
      const auto third_j = joint_path_derivative_state.third(j);
      const auto cond_j = !utils::isZero(third_j) && !utils::isZero(second_j) &&
                          !utils::isZero(first_j);

      if (!cond_j) {
        continue;
      }

      const auto jerk_abs_min_j = (first_j > 0.0)
                                      ? joint_constraints_.jerk_min(j)
                                      : joint_constraints_.jerk_max(j);

      const auto a = (third_j / first_j) - (third_i / first_i);
      const auto b = ((3.0 * second_j / first_j) - (3.0 * second_i / first_i)) *
                     path_state.acceleration;
      const auto c = (jerk_abs_max_i / first_i) - (jerk_abs_min_j / first_j);

      const auto bound_a = -4.0 * utils::pow(b, 3) / (27.0 * utils::pow(c, 2));

      if (a == 0.0) {
        continue;
      }

      Eigen::PolynomialSolver<double, Eigen::Dynamic> solver;
      Eigen::Vector4d coeff(c, b, 0.0, a);
      std::vector<double> real_roots(3, 0.0);
      solver.compute(coeff);
      solver.realRoots(real_roots);
      std::sort(real_roots.begin(), real_roots.end());

      if (c < 0) {
        if (b > 0) {
          if (bound_a < a && a < 0) {
            velocity_max = std::min(velocity_max, real_roots.at(2));
          }
        }
      } else {  // c > 0
        if (b < 0) {
          if (a < 0) {
            velocity_max = std::min(velocity_max, real_roots.at(0));
          } else if (0 < a && a < bound_a) {
            velocity_max = std::min(velocity_max, real_roots.at(1));
          }
        } else {  // b > 0
          if (a < bound_a) {
            velocity_max = std::min(velocity_max, real_roots.at(0));
          } else if (bound_a <= a && a < 0) {
            velocity_max = std::min(velocity_max, real_roots.at(2));
          }
        }
      }
    }
  }

  return std::make_pair(std::max(0.0, velocity_min), velocity_max);
}

}  // namespace rttopp
