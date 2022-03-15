#pragma once

#include <iostream>

#include <rttopp/path_acceleration_limits.h>
#include <rttopp/path_velocity_limit.h>
#include <rttopp/preprocessing.h>

namespace rttopp {

// TODO(wolfgang): try to increase MAX_WAYPOINTS and MAX_STEPS to generally
// usable defaults
// TODO(wolfgang): consider making MAX_WAYPOINTS and MAX_STEPS a regular
// constructor variable and use regular std vectors instead of static_vectors.
// Then the vector memory would be preallocated on the heap instead of the
// stack. On the one hand, we wouldn't be so much limited by the available stack
// memory. On the other hand, we could support two operating modes: a real-time
// mode with preallocation and fixed max sizes and a non-realtime mode where the
// maximum number of waypoints and integration steps doesn't need to be known
// beforehand. I'm not sure if the performance difference between heap and stack
// would be noticeable (static_vector vs std::vector with preallocation).
template <size_t N_JOINTS, size_t MAX_WAYPOINTS = 30, size_t MAX_STEPS = 6000>
class RTTOPP2 {
 public:
  explicit RTTOPP2(const double cycle_time = 0.001,
                   const double path_resolution = 0.05,
                   const size_t steps_per_cycle = 0)
      : cycle_time_(cycle_time),
        path_resolution_(path_resolution),
        steps_per_cycle_(steps_per_cycle),
        preprocess_(path_resolution, steps_per_cycle) {}

  // TODO(wolfgang): does a full parameterization, more methods are needed for a
  // per-cycle parameterization
  Result parameterizeFull(const Constraints<N_JOINTS> &constraints,
                          const Waypoints<N_JOINTS> &waypoints);

  [[nodiscard]] Result verifyTrajectory(
      bool verbose = false, size_t *num_idx = nullptr, double *mean = nullptr,
      double *std_dev = nullptr, double *max_normalized_velocity = nullptr,
      double *max_normalized_acceleration = nullptr) const;

  [[nodiscard]] nlohmann::json toJson() const;

 private:
  using JointPathDerivativeState = JointPathDerivatives<N_JOINTS>;

  Result passLocal(bool forward, bool first);
  [[nodiscard]] PathState integrateLocalForward(
      size_t current_idx, const PathState &current_state) const;
  [[nodiscard]] PathState integrateLocalBackward(
      size_t current_idx, const PathState &current_state) const;
  [[nodiscard]] double calculatePathAcceleration(
      const PathState &current_state, const PathState &previous_state) const;
  [[nodiscard]] WaypointJoint<N_JOINTS> calculateJointState(
      size_t current_idx, const PathState &current_state,
      const JointPathDerivativeState &joint_path_derivative_state) const;

  const double cycle_time_;
  const double path_resolution_;
  const double steps_per_cycle_;

  Constraints<N_JOINTS> constraints_;
  PathState start_state_, end_state_;
  size_t num_idx_;
  std::array<PathState, MAX_STEPS> backward_pass_states_;
  // TODO(wolfgang): rework so that second pass is final trajectory and not
  // saved internally
  std::array<PathState, MAX_STEPS> forward_pass_states_;
  std::array<WaypointJoint<N_JOINTS>, MAX_STEPS> joint_trajectory_;

  Preprocessing<N_JOINTS, MAX_WAYPOINTS, MAX_STEPS> preprocess_;

  // minimum velocity that needs to be enforced during integration
  // TODO(wolfgang): Leandro used 1.0e-02 in his tests, is this too low?
  static constexpr double MIN_VELOCITY = utils::EPS;
};

// TODO(wolfgang): pass a trajectory vector for forward param, internally only
// data for bw param should be stored and sampling should be default instead of
// full parameterization
template <size_t N_JOINTS, size_t MAX_WAYPOINTS, size_t MAX_STEPS>
Result RTTOPP2<N_JOINTS, MAX_WAYPOINTS, MAX_STEPS>::parameterizeFull(
    const Constraints<N_JOINTS> &constraints,
    const Waypoints<N_JOINTS> &waypoints) {
  bool forward_first = false;

  Result result;
  constraints_ = constraints;

  // TODO(wolfgang): clear param vectors or basically use them as arrays? boost
  // claims O(1)
  // resize is O(N)

  num_idx_ = preprocess_.processWaypoints(waypoints);

  start_state_.position = preprocess_.getPathPosition(0);
  start_state_.velocity = waypoints.front().joints.velocity.norm();
  end_state_.position = preprocess_.getPathPosition(num_idx_ - 1);
  end_state_.velocity = waypoints.back().joints.velocity.norm();

  if (forward_first) {
    result = passLocal(true, true);
    if (result.error()) {
      return result;
    }
    result = passLocal(false, false);
  } else {
    result = passLocal(false, true);
    if (result.error()) {
      return result;
    }
    result = passLocal(true, false);
  }

  // TODO(wolfgang): check here if start state valid and if target velocity
  // could be reached

  for (size_t idx = 0; idx < num_idx_; ++idx) {
    const auto &path_state =
        forward_first ? backward_pass_states_[idx] : forward_pass_states_[idx];
    joint_trajectory_[idx] =
        calculateJointState(idx, path_state, preprocess_.getDerivatives(idx));
  }

  return result;
}

// does not need to be optimized for second pass cause time integration will be
// used for that in the future
template <size_t N_JOINTS, size_t MAX_WAYPOINTS, size_t MAX_STEPS>
Result RTTOPP2<N_JOINTS, MAX_WAYPOINTS, MAX_STEPS>::passLocal(
    const bool forward, const bool first) {
  PathAccelerationLimits<N_JOINTS> path_acceleration_limits(
      constraints_.joints);
  PathVelocityLimit<N_JOINTS> path_velocity_limit(constraints_.joints);

  int current_idx = forward ? 0 : num_idx_ - 1;  // int cause we abort if < 0
  PathState current_state = forward ? start_state_ : end_state_;

  while (true) {
    if (forward) {
      if (current_idx >= int(num_idx_)) {
        break;
      }
    } else {
      if (current_idx < 0) {
        break;
      }
    }

    const auto joint_path_derivatives = preprocess_.getDerivatives(current_idx);

    if (first) {
      const auto vel_abs_max_first_order =
          path_velocity_limit.calculateJointVelocityLimit(
              joint_path_derivatives);

      current_state.velocity =
          std::min(current_state.velocity, vel_abs_max_first_order);
    } else {
      const auto &previous_pass_state = forward
                                            ? backward_pass_states_[current_idx]
                                            : forward_pass_states_[current_idx];
      if (current_state.velocity > previous_pass_state.velocity) {
        current_state.on_previous_limit_curve = true;
        current_state.velocity = previous_pass_state.velocity;
      }
    }

    double acc_min, acc_max;
    std::tie(acc_min, acc_max) =
        path_acceleration_limits.calculateDynamicLimits(current_state,
                                                        joint_path_derivatives);

    // TODO(wolfgang): this breaks the possibility to have the forward
    // integration as first run, but we can probably discard all the options
    // that enable the first run as forward integration. We will very likely not
    // need this option.
    const auto previous_state_on_mvc_second =
        (first && current_idx < int(num_idx_ - 1))
            ? backward_pass_states_[current_idx + 1].on_mvc_second
            : false;

    // either acceleration limits are invalid or they are at their max values
    // due to all first derivatives being zero in the first run
    // The previous state also needs to be checked that it's on the second-order
    // MVC. The reason is that there can be rare cases where the integration
    // accelerates from a point on the second-order MVC and immediately hits the
    // first-order MVC. It seems that in this case acc_min < acc_max. This means
    // that it's possible to get into a velocity area on the first-order MVC
    // where the acceleration ranges are valid, but it is actually above the
    // second-order MVC and above the area where the acceleration ranges are
    // invalid.
    // TODO(wolfgang): necessary to have this additional check in case that all
    // first derivatives are zero? Then min operator also needed for velocity
    // TODO(wolfgang): an alternative to all of these singularity avoidance
    // approaches below would be to limit the velocity to e.g. 90% of the
    // velocity on the second-order MVC (if it would be on there otherwise
    // during bw param). Then the acc_min > acc_max check cannot be used and we
    // would need to check the overall velocity limit in every bw param step in
    // the code above.
    if (acc_min > acc_max ||
        (first && acc_max <= utils::EPS && acc_min >= -utils::EPS) ||
        previous_state_on_mvc_second) {
      if (!first) {
        std::cout << "ERROR: should not happen, second pass and accelerations "
                     "invalid at idx "
                  << current_idx << std::endl;
        return Result::GENERAL_ERROR;
      }

      const auto vel_abs_max_second_order =
          path_velocity_limit.calculateJointAccelerationLimit(
              joint_path_derivatives);
      current_state.on_mvc_second =
          current_state.velocity > vel_abs_max_second_order ? true : false;
      current_state.velocity =
          std::min(current_state.velocity, vel_abs_max_second_order);

      std::tie(acc_min, acc_max) =
          path_acceleration_limits.calculateDynamicLimits(
              current_state, joint_path_derivatives);
    }
    assert(acc_min < acc_max);

    if ((forward && current_idx < int(num_idx_ - 1)) ||
        (!forward && current_idx > 0)) {
      current_state.acceleration = forward ? acc_max : acc_min;
      const auto &previous_state = forward
                                       ? forward_pass_states_[current_idx - 1]
                                       : backward_pass_states_[current_idx + 1];

      // A problem here with the bw pass is that the accel_min at the current
      // state is used for determining the velocity at the previous state.
      // However, the forward pass has only the accel at the previous state
      // available and thus, discrepancies can happen, mainly during braking the
      // forward pass has accel limit violations because the accel limits are
      // shiftet by one step. Solving this issue by looking ahead (one or two
      // steps) during the fw pass when braking and then switching to acc_min
      // did not work well. A lot of quick switches between acc_max and acc_min
      // happened because the fw pass was often quickly way below the bw limit
      // curve using acc_min and then switched back to acc_max. Limiting the fw
      // velocity to the min of the current and the next state actually led to
      // more accel limit violations because the fw pass had to brake harder.
      // Instead, the solution below is used where the bw accel is limited to
      // the max of acc_min at current pos and the next one
      if (!forward) {
        const auto next_state =
            integrateLocalBackward(current_idx, current_state);
        double acc_min_next, acc_max_next;
        std::tie(acc_min_next, acc_max_next) =
            path_acceleration_limits.calculateDynamicLimits(
                next_state, preprocess_.getDerivatives(current_idx - 1));
        if (acc_min_next < acc_max_next) {
          current_state.acceleration = std::max(acc_min, acc_min_next);
        } else {
          // The next state is above the second-order MVC, get away from the
          // singular points by choosing the maximum acceleration. This ensures
          // that the second pass has no instances where the velocity has to try
          // to move along the second-order MVC (which is generally not
          // possible).
          current_state.acceleration = std::max(acc_max, acc_max_next);
        }
        // This ensures that the second pass does not accelerate into a singular
        // point on the second-order MVC and instead brakes one step more to
        // stay away from the singularity. Both this and the additional braking
        // above should ensure that we do not get invalid accelerations in the
        // second pass around singularities.
      } else if (!first && (current_state.on_previous_limit_curve ||
                            previous_state.on_previous_limit_curve)) {
        const auto &next_state = forward
                                     ? backward_pass_states_[current_idx + 1]
                                     : forward_pass_states_[current_idx - 1];
        const auto &next_after_next_state =
            forward ? backward_pass_states_[current_idx + 2]
                    : forward_pass_states_[current_idx - 2];
        if (next_state.on_mvc_second || next_after_next_state.on_mvc_second) {
          current_state.acceleration = acc_min;
        }
      }
    } else {
      // set final state acceleration to zero to avoid confusion (if it's within
      // limits), otherwise to the value closest to zero
      if (acc_max > 0.0 && acc_min < 0.0) {
        current_state.acceleration = 0.0;
      } else {
        current_state.acceleration =
            std::abs(acc_max) < std::abs(acc_min) ? acc_max : acc_min;
      }
    }
    current_state.acc_max = acc_max;
    current_state.acc_min = acc_min;
    auto &parameterized_state = forward ? forward_pass_states_[current_idx]
                                        : backward_pass_states_[current_idx];
    parameterized_state = current_state;

    if (forward) {
      if (current_idx > 0) {
        auto &previous_state = forward_pass_states_[current_idx - 1];
        previous_state.acceleration =
            calculatePathAcceleration(current_state, previous_state);
      }

      if (current_idx < int(num_idx_ - 1)) {
        current_state = integrateLocalForward(current_idx, current_state);
      }
      current_idx++;
    } else {
      if (current_idx < int(num_idx_ - 1)) {
        // TODO(wolfgang): computes actual acceleration, only needed for
        // debugging/plotting in bw pass, can be moved
        auto &previous_state = backward_pass_states_[current_idx + 1];
        previous_state.acceleration =
            calculatePathAcceleration(current_state, previous_state);
      }

      if (current_idx > 0) {
        current_state = integrateLocalBackward(current_idx, current_state);
      }
      current_idx--;
    }
  }

  return Result::SUCCESS;
}

// TODO(wolfgang): this method will not be needed when switching to time
// integration for forward integration
template <size_t N_JOINTS, size_t MAX_WAYPOINTS, size_t MAX_STEPS>
PathState RTTOPP2<N_JOINTS, MAX_WAYPOINTS, MAX_STEPS>::integrateLocalForward(
    const size_t current_idx, const PathState &current_state) const {
  PathState next_state{};
  next_state.position = preprocess_.getPathPosition(current_idx + 1);
  const auto delta_position =
      next_state.position - preprocess_.getPathPosition(current_idx);
  const auto next_velocity =
      std::sqrt(utils::pow(current_state.velocity, 2) +
                2.0 * delta_position * current_state.acceleration);
  next_state.velocity = std::max(MIN_VELOCITY, next_velocity);

  return next_state;
}

template <size_t N_JOINTS, size_t MAX_WAYPOINTS, size_t MAX_STEPS>
PathState RTTOPP2<N_JOINTS, MAX_WAYPOINTS, MAX_STEPS>::integrateLocalBackward(
    const size_t current_idx, const PathState &current_state) const {
  PathState next_state{};
  next_state.position = preprocess_.getPathPosition(current_idx - 1);
  const auto delta_position =
      preprocess_.getPathPosition(current_idx) - next_state.position;
  const auto next_velocity =
      std::sqrt(utils::pow(current_state.velocity, 2) -
                2.0 * delta_position * current_state.acceleration);

  // TODO(wolfgang): minimum velocity enforced here. How to check if the
  // velocity stalls at this value and thus the parameterization is infeasible?
  // Barnett et al. recommend to check if it stalls for 5 time integration
  // steps. Their implementation just seems to check if a max time integration
  // steps bound is exceeded.
  next_state.velocity = std::max(MIN_VELOCITY, next_velocity);

  return next_state;
}

template <size_t N_JOINTS, size_t MAX_WAYPOINTS, size_t MAX_STEPS>
double RTTOPP2<N_JOINTS, MAX_WAYPOINTS, MAX_STEPS>::calculatePathAcceleration(
    const PathState &current_state, const PathState &previous_state) const {
  // see eq. 78 in Verscheure et al. Time-Optimal Path Tracking for Robots: A
  // Convex Optimization Approach, simply local integration formula reordered
  const auto delta_position = current_state.position - previous_state.position;
  assert(!utils::isZero(delta_position));

  const auto delta_velocity = utils::pow(current_state.velocity, 2) -
                              utils::pow(previous_state.velocity, 2);
  return delta_velocity / (2.0 * delta_position);
}

template <size_t N_JOINTS, size_t MAX_WAYPOINTS, size_t MAX_STEPS>
WaypointJoint<N_JOINTS>
RTTOPP2<N_JOINTS, MAX_WAYPOINTS, MAX_STEPS>::calculateJointState(
    const size_t current_idx, const PathState &current_state,
    const JointPathDerivativeState &joint_path_derivative_state) const {
  WaypointJoint<N_JOINTS> joint_state;
  joint_state.position = preprocess_.getJointPositionFromPath(current_idx);
  joint_state.velocity =
      joint_path_derivative_state.first * current_state.velocity;
  joint_state.acceleration =
      joint_path_derivative_state.second *
          utils::pow(current_state.velocity, 2) +
      joint_path_derivative_state.first * current_state.acceleration;

  return joint_state;
}

template <size_t N_JOINTS, size_t MAX_WAYPOINTS, size_t MAX_STEPS>
Result RTTOPP2<N_JOINTS, MAX_WAYPOINTS, MAX_STEPS>::verifyTrajectory(
    const bool verbose, size_t *num_idx, double *mean, double *std_dev,
    double *max_normalized_velocity,
    double *max_normalized_acceleration) const {
  PathVelocityLimit<N_JOINTS> path_velocity_limit(constraints_.joints);

  for (size_t idx = 0; idx < num_idx_; ++idx) {
    const auto joint_path_derivatives = preprocess_.getDerivatives(idx);
    const auto &bw_path_state = backward_pass_states_[idx];
    const auto &fw_path_state = forward_pass_states_[idx];
    const auto &joint_state = joint_trajectory_[idx];

    if (bw_path_state.velocity >
        path_velocity_limit.calculateOverallLimit(joint_path_derivatives)) {
      std::cout << "Error: bw path state velocity above MVC at idx " << idx
                << std::endl;
      return Result::GENERAL_ERROR;
    }
    if (fw_path_state.velocity > bw_path_state.velocity) {
      std::cout << "Error: fw path state velocity above bw path state velocity "
                   "at idx "
                << idx << std::endl;
      return Result::GENERAL_ERROR;
    }
    if (fw_path_state.acceleration > fw_path_state.acc_max + utils::EPS ||
        fw_path_state.acceleration < fw_path_state.acc_min - utils::EPS) {
      std::cout << "Error: fw path acceleration not within limits at idx "
                << idx << std::endl;
      std::cout << "limits: " << fw_path_state.acc_min << ", "
                << fw_path_state.acc_min << "; path acceleration "
                << fw_path_state.acceleration << std::endl;
      return Result::GENERAL_ERROR;
    }

    for (size_t joint = 0; joint < N_JOINTS; ++joint) {
      if (joint_state.velocity[joint] >
              constraints_.joints.velocity_max[joint] ||
          joint_state.velocity[joint] <
              constraints_.joints.velocity_min[joint]) {
        std::cout << "Error: joint velocity not within limits at idx " << idx
                  << " and joint " << joint << std::endl;
        return Result::GENERAL_ERROR;
      }
      if (joint_state.acceleration[joint] >
              constraints_.joints.acceleration_max[joint] ||
          joint_state.acceleration[joint] <
              constraints_.joints.acceleration_min[joint]) {
        std::cout << "Error: joint acceleration not within limits at idx "
                  << idx << " and joint " << joint << std::endl;
        return Result::GENERAL_ERROR;
      }
    }
  }

  // TODO(wolfgang): When time integration is ready, also take the time
  // difference of the position signal and check if its within limits?

  double max_normalized_vel = 0.0, max_normalized_accel = 0.0;
  std::vector<WaypointJoint<N_JOINTS>> joint_trajectory_normalized(num_idx_);
  Eigen::VectorXd infinity_norm(num_idx_);
  for (size_t idx = 0; idx < num_idx_; ++idx) {
    const auto &joint_state = joint_trajectory_[idx];
    auto &joint_state_normalized = joint_trajectory_normalized[idx];

    for (size_t joint = 0; joint < N_JOINTS; ++joint) {
      // TODO(wolfgang): taking the max only works when the limits have
      // different signs. When the signs are the same, the min has to be taken.
      joint_state_normalized.velocity[joint] = std::max(
          joint_state.velocity[joint] / constraints_.joints.velocity_max[joint],
          joint_state.velocity[joint] /
              constraints_.joints.velocity_min[joint]);
      joint_state_normalized.acceleration[joint] =
          std::max(joint_state.acceleration[joint] /
                       constraints_.joints.acceleration_max[joint],
                   joint_state.acceleration[joint] /
                       constraints_.joints.acceleration_min[joint]);
      if (joint_state_normalized.velocity[joint] > max_normalized_vel) {
        max_normalized_vel = joint_state_normalized.velocity[joint];
      }
      if (joint_state_normalized.acceleration[joint] > max_normalized_accel) {
        max_normalized_accel = joint_state_normalized.acceleration[joint];
      }
    }

    infinity_norm[idx] =
        std::max(joint_state_normalized.velocity.maxCoeff(),
                 joint_state_normalized.acceleration.maxCoeff());
  }

  if (num_idx != nullptr) {
    *num_idx = num_idx_;
  }
  // TODO(wolfgang): enforce a certain mean and std_dev, e.g. > 90% for mean and
  // < 0.1 for std_dev?
  double mean_local = infinity_norm.mean();
  if (mean != nullptr) {
    *mean = mean_local;
  }
  double std_dev_local =
      std::sqrt((infinity_norm.array() - mean_local).square().sum() /
                double(infinity_norm.size() - 1));
  if (std_dev != nullptr) {
    *std_dev = std_dev_local;
  }
  if (max_normalized_velocity != nullptr) {
    *max_normalized_velocity = max_normalized_vel;
  }
  if (max_normalized_acceleration != nullptr) {
    *max_normalized_acceleration = max_normalized_accel;
  }

  if (verbose) {
    std::cout << "trajectory evaluation data:" << std::endl;
    std::cout << "number of nodes: " << num_idx_ << std::endl;
    std::cout << "max normalized velocity " << max_normalized_vel
              << ", max normalized acceleration " << max_normalized_accel
              << std::endl;
    std::cout << "infinity norm of velocities and accelerations: mean "
              << mean_local << ", standard deviation " << std_dev_local
              << std::endl
              << std::endl;
  }

  return Result::SUCCESS;
}

template <size_t N_JOINTS, size_t MAX_WAYPOINTS, size_t MAX_STEPS>
nlohmann::json RTTOPP2<N_JOINTS, MAX_WAYPOINTS, MAX_STEPS>::toJson() const {
  nlohmann::json j;
  PathVelocityLimit<N_JOINTS> path_velocity_limit(constraints_.joints);

  for (size_t idx = 0; idx < num_idx_; ++idx) {
    j["path_parameterization"][idx]["idx"] = idx;

    const auto &forward_state = forward_pass_states_[idx];
    j["path_parameterization"][idx]["position"] = forward_state.position;
    j["path_parameterization"][idx]["forward"]["velocity"] =
        forward_state.velocity;
    j["path_parameterization"][idx]["forward"]["acceleration"] =
        forward_state.acceleration;
    j["path_parameterization"][idx]["acc_min"] = forward_state.acc_min;
    j["path_parameterization"][idx]["acc_max"] = forward_state.acc_max;

    const auto &backward_state = backward_pass_states_[idx];
    j["path_parameterization"][idx]["backward"]["velocity"] =
        backward_state.velocity;
    j["path_parameterization"][idx]["backward"]["acceleration"] =
        backward_state.acceleration;
    j["path_parameterization"][idx]["backward"]["acc_min"] =
        backward_state.acc_min;
    j["path_parameterization"][idx]["backward"]["acc_max"] =
        backward_state.acc_max;

    // data, that we need for phase space plotting and that is not directly
    // computed here: absolute MVC (first-order + second-order)
    const auto joint_path_derivatives = preprocess_.getDerivatives(idx);
    // also save derivatives for debugging
    std::vector<double> derivatives_first, derivatives_second,
        derivatives_third;
    utils::setMatrixAsVector(derivatives_first, joint_path_derivatives.first);
    utils::setMatrixAsVector(derivatives_second, joint_path_derivatives.second);
    utils::setMatrixAsVector(derivatives_third, joint_path_derivatives.third);
    j["path_parameterization"][idx]["derivatives"]["first"] = derivatives_first;
    j["path_parameterization"][idx]["derivatives"]["second"] =
        derivatives_second;
    j["path_parameterization"][idx]["derivatives"]["third"] = derivatives_third;
    j["path_parameterization"][idx]["vel_abs_max"] =
        path_velocity_limit.calculateOverallLimit(joint_path_derivatives);
    j["path_parameterization"][idx]["vel_abs_max_first"] =
        path_velocity_limit.calculateJointVelocityLimit(joint_path_derivatives);
    j["path_parameterization"][idx]["vel_abs_max_second"] =
        path_velocity_limit.calculateJointAccelerationLimit(
            joint_path_derivatives);

    // TODO(wolfgang): set same joint constraints for every idx. When
    // considering dynamics, replace these with computed constraints
    std::vector<double> joint_acceleration_min;
    utils::setMatrixAsVector(joint_acceleration_min,
                             constraints_.joints.acceleration_min);
    j["path_parameterization"][idx]["joint_constraints"]["acc_min"] =
        joint_acceleration_min;

    std::vector<double> joint_acceleration_max;
    utils::setMatrixAsVector(joint_acceleration_max,
                             constraints_.joints.acceleration_max);
    j["path_parameterization"][idx]["joint_constraints"]["acc_max"] =
        joint_acceleration_max;

    j["path_parameterization"][idx]["jointtrajpoints"]["idx"] = idx;
    j["path_parameterization"][idx]["jointtrajpoints"]["values"] =
        utils::jointStateToJson(joint_trajectory_[idx]);
  }

  std::vector<double> joint_velocity_min;
  utils::setMatrixAsVector(joint_velocity_min,
                           constraints_.joints.velocity_min);
  j["joint_constraints"]["vel_min"] = joint_velocity_min;

  std::vector<double> joint_velocity_max;
  utils::setMatrixAsVector(joint_velocity_max,
                           constraints_.joints.velocity_max);
  j["joint_constraints"]["vel_max"] = joint_velocity_max;

  return j;
}

}  // namespace rttopp
