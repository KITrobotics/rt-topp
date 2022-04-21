#pragma once

#include <rttopp/types_utils.h>

namespace rttopp {

template <size_t N_JOINTS>
class PathAccelerationLimits {
  using JointPathDerivativeState = JointPathDerivatives<N_JOINTS>;

 public:
  explicit PathAccelerationLimits(
      const JointConstraints<N_JOINTS> &joint_constraints)
      : joint_constraints_(joint_constraints) {}

  /**
   * @brief Calculates overall path acceleration limits when
   * considering joint velocity, acceleration and jerk constraints.
   * Internally calls calculateJointVelocityLimits,
   * calculateJointAccelerationLimits and calculateJointJerkLimits
   *
   * @param path_state
   * @param joint_path_derivative_state
   *
   * @return Minimum and maximum overall acceleration limits
   * respectively
   */
  [[nodiscard]] std::pair<double, double> calculateOverallLimits(
      const PathState &path_state,
      const JointPathDerivativeState &joint_path_derivative_state);

  /**
   * @brief Calculates path acceleration limits based only on joint acceleration
   * constraints
   *
   * @param path_state
   * @param joint_path_derivative_state
   *
   * @return Minimum and maximum path acceleration limits respectively when
   * considering only joint acceleration constraints
   */
  [[nodiscard]] std::pair<double, double> calculateJointAccelerationLimits(
      const PathState &path_state,
      const JointPathDerivativeState &joint_path_derivative_state);

  /**
   * @brief Calculates path acceleration limits based only on joint jerk
   * constraints
   *
   * @param path_state
   * @param joint_path_derivative_state
   *
   * @return Minimum and maximum path acceleration limits respectively when
   * considering only joint jerk constraints
   */
  [[nodiscard]] std::pair<double, double> calculateJointJerkLimits(
      const PathState &path_state,
      const JointPathDerivativeState &joint_path_derivative_state);

  /**
   * @brief Calculates the min and max acceleration profiles according to the
   * classic TOPP formulation depending on the current path state (see for
   * example Pham et al 2014 - A General, Fast, and Robust Implementation of the
   * Time-Optimal Path Parameterization Algorithm). Please note that with this
   * definition min acceleration may be eventually greater than max acceleration
   * and this property is used to find acceleration switching points
   *
   * @param path_state The path position and its derivatives at a given point
   * @param joint_path_derivative_state The joint positions and their time
   * derivatives wrt path
   *
   * @return The minimum and maximum acceleration values for the queried state
   * respectively
   */
  [[nodiscard]] std::pair<double, double> calculateDynamicLimits(
      const PathState &path_state,
      const JointPathDerivativeState &joint_path_derivative_state);

 private:
  /**
   * @brief Calculates path acceleration limits based only on joint acceleration
   * constraints. This method handles the case for which both qs and qss are not
   * equal to zero
   *
   * @param path_state
   * @param joint_path_derivative_state
   *
   * @return Minimum and maximum path acceleration limits respectively
   */
  [[nodiscard]] std::pair<double, double> calculateJointAccelerationLimits1(
      const PathState &path_state,
      const JointPathDerivativeState &joint_path_derivative_state);

  /**
   * @brief Calculates path acceleration limits based only on joint acceleration
   * constraints. This method handles the case for which both qss is equal to
   * zero
   *
   * @param path_state
   * @param joint_path_derivative_state
   *
   * @return Minimum and maximum path acceleration limits respectively
   */
  [[nodiscard]] std::pair<double, double> calculateJointAccelerationLimits2(
      const PathState &path_state,
      const JointPathDerivativeState &joint_path_derivative_state);

  [[nodiscard]] std::pair<double, double> calculateJointJerkLimits1(
      const PathState &path_state,
      const JointPathDerivativeState &joint_path_derivative_state);
  [[nodiscard]] std::pair<double, double> calculateJointJerkLimits2(
      const PathState &path_state,
      const JointPathDerivativeState &joint_path_derivative_state);
  [[nodiscard]] std::pair<double, double> calculateJointJerkLimits3(
      const PathState &path_state,
      const JointPathDerivativeState &joint_path_derivative_state);

  const JointConstraints<N_JOINTS> &joint_constraints_;
};

template <size_t N_JOINTS>
std::pair<double, double>
PathAccelerationLimits<N_JOINTS>::calculateDynamicLimits(
    const PathState &path_state,
    const JointPathDerivativeState &joint_path_derivative_state) {
  double min_acceleration = std::numeric_limits<double>::lowest();
  double max_acceleration = std::numeric_limits<double>::max();

  for (std::size_t i = 0; i < N_JOINTS; ++i) {
    const auto first_i = joint_path_derivative_state.first(i);
    const auto second_i = joint_path_derivative_state.second(i);

    const auto a_max = first_i;
    const auto b_max = second_i;
    const auto c_max = -joint_constraints_.acceleration_max(i);

    const auto a_min = -a_max;
    const auto b_min = -b_max;
    const auto c_min = joint_constraints_.acceleration_min(i);

    // TODO(wolfgang): should we have a check here when a_() is zero? might get
    // NaN or other issues otherwise.
    if (utils::isZero(a_max)) {
      continue;
    }

    const auto with_joint_acceleration_max =
        (-c_max - b_max * utils::pow(path_state.velocity, 2)) / a_max;

    const auto with_joint_acceleration_min =
        (-c_min - b_min * utils::pow(path_state.velocity, 2)) / a_min;

    if (a_max < 0.0) {
      min_acceleration =
          std::max(min_acceleration, with_joint_acceleration_max);
    } else if (a_max > 0.0) {
      max_acceleration =
          std::min(max_acceleration, with_joint_acceleration_max);
    }

    if (a_min < 0.0) {
      min_acceleration =
          std::max(min_acceleration, with_joint_acceleration_min);
    } else if (a_min > 0.0) {
      max_acceleration =
          std::min(max_acceleration, with_joint_acceleration_min);
    }
  }

  // at least one first derivative should not be zero
  assert(min_acceleration > std::numeric_limits<double>::lowest() &&
         max_acceleration < std::numeric_limits<double>::max());

  // stay a little inside the limits
  return std::make_pair(min_acceleration + utils::EPS,
                        max_acceleration - utils::EPS);
}
}  // namespace rttopp
