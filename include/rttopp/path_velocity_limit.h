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
      velocity_max = std::min(
          velocity_max,
          std::sqrt(joint_constraints_.acceleration_min[i] / second_i));
    } else if (second_i > 0.0 && joint_constraints_.acceleration_max[i] > 0.0) {
      velocity_max = std::min(
          velocity_max,
          std::sqrt(joint_constraints_.acceleration_max[i] / second_i));
    }
  }

  return velocity_max;
}

}  // namespace rttopp
