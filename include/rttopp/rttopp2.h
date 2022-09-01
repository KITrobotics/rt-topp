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
      : preprocess_(path_resolution, steps_per_cycle),
        cycle_time_(cycle_time),
        path_resolution_(path_resolution),
        steps_per_cycle_(steps_per_cycle) {}

  // TODO(wolfgang): public methods do no yet check for invalid input (e.g. all
  // numbers finite, constraints vel_min < 0, vel_max > 0, min < max,
  // cycle_time/path_resolution > 0.0)

  Result parameterizeFull(const Constraints<N_JOINTS> &constraints,
                          const Waypoints<N_JOINTS> &waypoints);

  // per-cycle methods
  Result initParameterization(const Constraints<N_JOINTS> &constraints,
                              const Waypoints<N_JOINTS> &waypoints);
  Result keepParam();
  Result sample(State<N_JOINTS> *new_state);

  // generate whole trajectory at once, not real-time
  // needs at least a call to initParameterization() first
  Result sampleFull(Trajectory<N_JOINTS> *trajectory);

  [[nodiscard]] Result verifyTrajectory(
      const Trajectory<N_JOINTS> &trajectory, bool verbose = false,
      size_t *num_idx = nullptr, double *mean = nullptr,
      double *std_dev = nullptr, double *max_normalized_velocity = nullptr,
      double *max_normalized_acceleration = nullptr) const;
  [[nodiscard]] Result verifyTrajectory(
      bool verbose = false, size_t *num_idx = nullptr, double *mean = nullptr,
      double *std_dev = nullptr, double *max_normalized_velocity = nullptr,
      double *max_normalized_acceleration = nullptr) const;

  [[nodiscard]] nlohmann::json toJson(
      const Waypoints<N_JOINTS> &waypoints,
      const Trajectory<N_JOINTS> &trajectory) const;
  [[nodiscard]] nlohmann::json toJson(
      const Waypoints<N_JOINTS> &waypoints) const;

 private:
  using JointPathDerivativeState = JointPathDerivatives<N_JOINTS>;

  Result passLocal(bool forward, bool first);
  [[nodiscard]] PathState integrateLocalForward(
      size_t current_idx, const PathState &current_state) const;
  [[nodiscard]] PathState integrateLocalBackward(
      size_t current_idx, const PathState &current_state) const;
  [[nodiscard]] double calculatePathAcceleration(
      const PathState &current_state, const PathState &previous_state) const;
  [[nodiscard]] WaypointJoint<N_JOINTS> calculateJointDerivativeState(
      const PathState &current_state,
      const JointPathDerivativeState &joint_path_derivative_state) const;

  Preprocessing<N_JOINTS, MAX_WAYPOINTS, MAX_STEPS> preprocess_;

  Constraints<N_JOINTS> constraints_;

  // TODO(wolfgang): rework so that second pass is final trajectory and not
  // saved internally
  std::array<WaypointJoint<N_JOINTS>, MAX_STEPS> joint_trajectory_;

  const double cycle_time_;
  const double path_resolution_;
  const double steps_per_cycle_;

  size_t num_idx_;
  PathState start_state_, end_state_;
  std::array<PathState, MAX_STEPS> backward_pass_states_;
  // TODO(wolfgang): rework so that second pass is final trajectory and not
  // saved internally
  std::array<PathState, MAX_STEPS> forward_pass_states_;

  State<N_JOINTS> current_full_state_;

  // minimum velocity that needs to be enforced during integration
  static constexpr double MIN_VELOCITY = utils::EPS;
  // tolerance for acceleration limit violations
  static constexpr double ACCELERATION_TOLERANCE = utils::EPS * 1000;
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

  for (size_t idx = 0; idx < num_idx_; ++idx) {
    const auto &path_state =
        forward_first ? backward_pass_states_[idx] : forward_pass_states_[idx];
    joint_trajectory_[idx] = calculateJointDerivativeState(
        path_state, preprocess_.getDerivatives(idx));
    joint_trajectory_[idx].position = preprocess_.getJointPositionFromPath(idx);
  }

  return result;
}

template <size_t N_JOINTS, size_t MAX_WAYPOINTS, size_t MAX_STEPS>
Result RTTOPP2<N_JOINTS, MAX_WAYPOINTS, MAX_STEPS>::initParameterization(
    const Constraints<N_JOINTS> &constraints,
    const Waypoints<N_JOINTS> &waypoints) {
  constraints_ = constraints;

  // TODO(wolfgang): does a full bw param for now

  num_idx_ = preprocess_.processWaypoints(waypoints);

  start_state_.position = preprocess_.getPathPosition(0);
  start_state_.velocity = waypoints.front().joints.velocity.norm();
  end_state_.position = preprocess_.getPathPosition(num_idx_ - 1);
  end_state_.velocity = waypoints.back().joints.velocity.norm();

  Result result = passLocal(false, true);
  if (result.error()) {
    return result;
  }

  current_full_state_.path_state = start_state_;
  current_full_state_.bw_idx = 0;
  current_full_state_.time = 0.0;

  return result;
}

template <size_t N_JOINTS, size_t MAX_WAYPOINTS, size_t MAX_STEPS>
Result RTTOPP2<N_JOINTS, MAX_WAYPOINTS, MAX_STEPS>::keepParam() {
  // TODO(wolfgang): implement this and change initParameterization()

  return Result::SUCCESS;
}

template <size_t N_JOINTS, size_t MAX_WAYPOINTS, size_t MAX_STEPS>
Result RTTOPP2<N_JOINTS, MAX_WAYPOINTS, MAX_STEPS>::sampleFull(
    Trajectory<N_JOINTS> *trajectory) {
  Result result;

  do {
    State<N_JOINTS> new_state;
    result = sample(&new_state);
    trajectory->push_back(new_state);
  } while (result.val() == Result::WORKING);

  return result;
}

template <size_t N_JOINTS, size_t MAX_WAYPOINTS, size_t MAX_STEPS>
Result RTTOPP2<N_JOINTS, MAX_WAYPOINTS, MAX_STEPS>::sample(
    State<N_JOINTS> *new_state) {
  PathAccelerationLimits<N_JOINTS> path_acceleration_limits(
      constraints_.joints);
  PathVelocityLimit<N_JOINTS> path_velocity_limit(constraints_.joints);

  auto &path_state = current_full_state_.path_state;
  auto &derivatives = current_full_state_.derivatives;

  if (path_state.position >= end_state_.position) {
    // TODO(wolfgang): lineary extrapolate here with constant velocity in
    // preprocessing if goal velocity != 0?

    if (path_state.velocity < end_state_.velocity) {
      return Result::GOAL_STATE_VELOCITY_TOO_HIGH;
    }

    return Result::FINISHED;
  }

  const auto &previous_pass_state =
      backward_pass_states_[current_full_state_.bw_idx];

  double acc_min, acc_max;
  std::tie(acc_min, acc_max) =
      path_acceleration_limits.calculateDynamicLimits(path_state, derivatives);
  if (acc_min > acc_max) {
    std::cout << "above MVC with velocity " << path_state.velocity
              << " at position " << path_state.position << ", idx "
              << current_full_state_.bw_idx << ", prev pass vel "
              << previous_pass_state.velocity << ", prev pass pos "
              << previous_pass_state.position << " and time "
              << current_full_state_.time << std::endl;
    std::cout << "derivs first " << derivatives.first << std::endl;
    std::cout << "bw derivs first "
              << preprocess_.getDerivatives(current_full_state_.bw_idx).first
              << std::endl;
    std::cout << "derivs second " << derivatives.second << std::endl;
    std::cout << "bw derivs second "
              << preprocess_.getDerivatives(current_full_state_.bw_idx).second
              << std::endl;
  }
  assert(acc_min < acc_max);
  // TODO(wolfgang): if final state, use passLocal() clipping
  path_state.acceleration = acc_max;
  path_state.acc_max = acc_max;
  path_state.acc_min = acc_min;

  double old_path_vel = path_state.velocity;
  path_state.velocity += path_state.acceleration * cycle_time_;

  // TODO(wolfgang): move formula to own method
  double position_diff =
      0.5 * (old_path_vel + path_state.velocity) * cycle_time_;

  assert(current_full_state_.bw_idx + 1 < num_idx_);
  /* std::cout << "before increasing sample idx by one, bw idx " */
  /*           << current_full_state_.bw_idx << ", bw pos " */
  /*           << backward_pass_states_[current_full_state_.bw_idx].position */
  /*           << ", old pos " << path_state.position << ", pos diff " */
  /*           << position_diff << ", time " << current_full_state_.time */
  /*           << std::endl; */

  while (backward_pass_states_[current_full_state_.bw_idx].position <
         path_state.position + position_diff) {
    current_full_state_.bw_idx++;

    if (current_full_state_.bw_idx == num_idx_) {
      // we are finished, reached goal position
      current_full_state_.bw_idx--;
      path_state.position =
          backward_pass_states_[current_full_state_.bw_idx].position;
      // TODO(wolfgang): lineary extrapolate if velocity > 0
      path_state.velocity =
          std::min(path_state.velocity,
                   backward_pass_states_[current_full_state_.bw_idx].velocity);
      path_state.acceleration =
          (path_state.velocity - old_path_vel) / cycle_time_;

      current_full_state_.waypoint.joints = calculateJointDerivativeState(
          backward_pass_states_[current_full_state_.bw_idx],
          preprocess_.getDerivatives(current_full_state_.bw_idx));
      current_full_state_.waypoint.joints.position =
          preprocess_.getJointPositionFromPath(current_full_state_.bw_idx);
      current_full_state_.time += cycle_time_;

      *new_state = current_full_state_;

      if (path_state.velocity < end_state_.velocity) {
        return Result::GOAL_STATE_VELOCITY_TOO_HIGH;
      }

      return Result::FINISHED;
    }

    /* std::cout << "increased sample idx by one, bw idx " */
    /*           << current_full_state_.bw_idx << ", bw pos " */
    /*           << backward_pass_states_[current_full_state_.bw_idx].position
     */
    /*           << ", old pos " << path_state.position << ", pos diff " */
    /*           << position_diff << ", time " << current_full_state_.time */
    /*           << std::endl; */
    assert(current_full_state_.bw_idx > 0 &&
           current_full_state_.bw_idx < num_idx_);
  }
  const auto &next_pass_state =
      backward_pass_states_[current_full_state_.bw_idx];
  // linear interpolation between next and previous pass state
  const auto interp_next_vel =
      previous_pass_state.velocity +
      (path_state.position - previous_pass_state.position) /
          (next_pass_state.position - previous_pass_state.position) *
          (next_pass_state.velocity - previous_pass_state.velocity);

  if (interp_next_vel < path_state.velocity) {
    std::cout << "limited to interp vel " << interp_next_vel << " at time "
              << current_full_state_.time << " and old position "
              << path_state.position << std::endl;
    // clip again and integrate again
    path_state.velocity = interp_next_vel;
    position_diff = 0.5 * (old_path_vel + path_state.velocity) * cycle_time_;
  }

  if (backward_pass_states_[current_full_state_.bw_idx - 1].position >
      path_state.position + position_diff) {
    std::cout
        << "ERROR: increased bw idx too far, should go back again!! bw idx "
        << current_full_state_.bw_idx << ", position "
        << backward_pass_states_[current_full_state_.bw_idx].position
        << ", new pos " << path_state.position + position_diff << std::endl;
    do {
      current_full_state_.bw_idx--;
      assert(current_full_state_.bw_idx > 0 &&
             current_full_state_.bw_idx < num_idx_);
    } while (backward_pass_states_[current_full_state_.bw_idx - 1].position >
             path_state.position + position_diff);

    const auto &new_next_pass_state =
        backward_pass_states_[current_full_state_.bw_idx];
    if (new_next_pass_state.velocity < path_state.velocity) {
      std::cout << "need to clip again after goint back" << std::endl;
      // clip again and integrate again
      path_state.velocity = next_pass_state.velocity;
      position_diff = 0.5 * (old_path_vel + path_state.velocity) * cycle_time_;
    }
  }

  // TODO(wolfgang): egg, hen problem, needs to be recalculated if pos changes
  auto joint_pos_derivatives_pair =
      preprocess_.computeJointPositionAndDerivatives(path_state.position +
                                                     position_diff);
  derivatives = joint_pos_derivatives_pair.second;
  const auto vel_abs_max =
      path_velocity_limit.calculateOverallLimit(derivatives);
  if (vel_abs_max < path_state.velocity) {
    path_state.velocity = vel_abs_max;
    position_diff = 0.5 * (old_path_vel + path_state.velocity) * cycle_time_;

    std::cout << "limited to MVC vel " << vel_abs_max << " at time "
              << current_full_state_.time << " and old position "
              << path_state.position << std::endl;
  }

  if (old_path_vel <= MIN_VELOCITY && path_state.velocity <= MIN_VELOCITY) {
    std::cout << "stalling at time " << current_full_state_.time * 1000
              << " ms and idx " << current_full_state_.bw_idx << ", num idx "
              << num_idx_ << ", bw pos "
              << backward_pass_states_[current_full_state_.bw_idx].position
              << ", old pos " << path_state.position << ", pos diff "
              << position_diff << std::endl;
    return Result::NOT_SOLVABLE_VELOCITY_STALLING;
  }

  path_state.position += position_diff;
  path_state.acceleration = (path_state.velocity - old_path_vel) / cycle_time_;
  // decrease again to be before
  current_full_state_.bw_idx--;

  current_full_state_.waypoint.joints =
      calculateJointDerivativeState(path_state, derivatives);
  current_full_state_.waypoint.joints.position =
      joint_pos_derivatives_pair.first;
  current_full_state_.time += cycle_time_;

  *new_state = current_full_state_;

  return Result::WORKING;
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
      // TODO(wolfgang): Since the changes on the wolfgang_clean_mvc2stayaway
      // and wolfgang_clean_fixMVC branches, the absolute value is taken here.
      // The acc_min > acc_max check is completely removed and thus,
      // second-order MVC values are computed in every bw step. This allowed to
      // use a velocity factor for the MVC2 value, but it significantly
      // increases the overall computation time. The check could be reintroduced
      // if it turns out that using it does not cause instabilities. Using the
      // check alone is problematic because the bw
      // integration can reach areas above MVC2 where acc_min < acc_max again
      // and thus this check alone is not sufficient or unstable. The velocity
      // factor for MVC2 is currently not needed. If it is needed again, it
      // should be used here and not in the PathVelocityLimit class. The value
      // in the class should be the exact one and if that value turns out to not
      // be controllable here in the algorithm, a factor should be applied here.

      const auto vel_max_first =
          path_velocity_limit.calculateJointVelocityLimit(
              joint_path_derivatives);
      const auto vel_max_second =
          path_velocity_limit.calculateJointAccelerationLimit(
              joint_path_derivatives);
      const auto vel_abs_max = std::min(vel_max_first, vel_max_second);

      if (current_state.velocity > vel_max_second &&
          vel_max_second < vel_max_first) {
        // Check if forward integrating with singular acceleration produces a
        // negative velocity. Then the velocity needs to be limited to zero
        // velocity in this step.
        current_state.velocity = vel_max_second;
        double acc_min, acc_max;
        std::tie(acc_min, acc_max) =
            path_acceleration_limits.calculateDynamicLimits(
                current_state, joint_path_derivatives);
        current_state.acceleration = acc_max;
        const auto next_state =
            integrateLocalForward(current_idx, current_state);
        if (next_state.velocity <= MIN_VELOCITY) {
          current_state.velocity = MIN_VELOCITY;
        }
      }

      current_state.velocity = std::min(current_state.velocity, vel_abs_max);
    } else {
      const auto &previous_pass_state = forward
                                            ? backward_pass_states_[current_idx]
                                            : forward_pass_states_[current_idx];
      current_state.velocity =
          std::min(current_state.velocity, previous_pass_state.velocity);
    }

    // these two checks below ensure that start and goal state are below the MVC
    // at the end of passLocal(), it is checked if the states are actually
    // reached
    if (!forward && current_idx == int(num_idx_ - 1) &&
        current_state.velocity < end_state_.velocity) {
      return Result::GOAL_STATE_VELOCITY_TOO_HIGH;
    }

    if (forward && current_idx == 0 &&
        current_state.velocity < start_state_.velocity) {
      /* std::cout << "start above MVC, vel current " << current_state.velocity
       * << ", start " << start_state_.velocity << std::endl; */
      return Result::START_STATE_VELOCITY_TOO_HIGH;
    }

    if (current_idx < int(num_idx_ - 1) && current_idx > 0) {
      const auto &previous_state = forward
                                       ? forward_pass_states_[current_idx - 1]
                                       : backward_pass_states_[current_idx + 1];
      // When velocity stalls at the minimum value over two
      // positions, the problem is not solvable
      if (current_state.velocity <= MIN_VELOCITY &&
          previous_state.velocity <= MIN_VELOCITY) {
        /* std::cout << "velocity stalling at idx " << current_idx << " in run "
         */
        /* << (forward ? "forward" : "backward") << std::endl; */
        return Result::NOT_SOLVABLE_VELOCITY_STALLING;
      }
    }

    double acc_min, acc_max;
    std::tie(acc_min, acc_max) =
        path_acceleration_limits.calculateDynamicLimits(current_state,
                                                        joint_path_derivatives);
    assert(acc_min < acc_max);

    if ((forward && current_idx < int(num_idx_ - 1)) ||
        (!forward && current_idx > 0)) {
      current_state.acceleration = forward ? acc_max : acc_min + utils::EPS;

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
        auto next_state = integrateLocalBackward(current_idx, current_state);
        const auto next_derivatives =
            preprocess_.getDerivatives(current_idx - 1);
        // ensure that the next state is valid and below or on the MVC
        const auto vel_abs_max =
            path_velocity_limit.calculateOverallLimit(next_derivatives);
        next_state.velocity = std::min(next_state.velocity, vel_abs_max);

        double acc_min_next, acc_max_next;
        std::tie(acc_min_next, acc_max_next) =
            path_acceleration_limits.calculateDynamicLimits(next_state,
                                                            next_derivatives);
        assert(acc_min_next < acc_max_next);
        // add a little eps for the bw accel to avoid little numerical
        // discrepancies between fw and bw param
        current_state.acceleration =
            std::max(acc_min, acc_min_next) + utils::EPS;
      }
      // TODO(wolfgang): several steps for the forward integration were removed
      // here on the wolfgang_clean_mvc2stayaway branch. They allowed to brake a
      // few steps further to avoid MVC2 singularities. They were generally
      // useful and could be reintroduced in the time integration variant
      // (instead of e.g. reducing the velocity factor for the MVC2 value which
      // has a much larger negative impact on the optimality.)
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
        auto &previous_fw_state = forward_pass_states_[current_idx - 1];
        previous_fw_state.acceleration =
            calculatePathAcceleration(current_state, previous_fw_state);
      }

      if (current_idx < int(num_idx_ - 1)) {
        current_state = integrateLocalForward(current_idx, current_state);
      }
      current_idx++;
    } else {
      if (current_idx < int(num_idx_ - 1)) {
        // TODO(wolfgang): computes actual acceleration, only needed for
        // debugging/plotting in bw pass, can be moved
        auto &previous_bw_state = backward_pass_states_[current_idx + 1];
        previous_bw_state.acceleration =
            calculatePathAcceleration(current_state, previous_bw_state);
      }

      if (current_idx > 0) {
        current_state = integrateLocalBackward(current_idx, current_state);
      }
      current_idx--;
    }
  }

  if (!forward &&
      start_state_.velocity > backward_pass_states_.front().velocity) {
    /* std::cout << "start_state " << start_state_.velocity << ", " << */
    /* backward_pass_states_.front().velocity << std::endl; */
    return Result::START_STATE_VELOCITY_TOO_HIGH;
  }

  if (forward &&
      end_state_.velocity > forward_pass_states_[num_idx_ - 1].velocity) {
    /* std::cout << "end_state " << end_state_.velocity << ", " <<
     * forward_pass_states_[num_idx_ - 1].velocity << std::endl; */
    return Result::GOAL_STATE_VELOCITY_TOO_HIGH;
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
  assert(!utils::isZero(delta_position));
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
  assert(!utils::isZero(delta_position));
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
RTTOPP2<N_JOINTS, MAX_WAYPOINTS, MAX_STEPS>::calculateJointDerivativeState(
    const PathState &current_state,
    const JointPathDerivativeState &joint_path_derivative_state) const {
  WaypointJoint<N_JOINTS> joint_state;
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
    const Trajectory<N_JOINTS> &trajectory, const bool verbose, size_t *num_idx,
    double *mean, double *std_dev, double *max_normalized_velocity,
    double *max_normalized_acceleration) const {
  PathVelocityLimit<N_JOINTS> path_velocity_limit(constraints_.joints);

  for (size_t idx = 0; idx < num_idx_; ++idx) {
    const auto joint_path_derivatives = preprocess_.getDerivatives(idx);
    const auto &bw_path_state = backward_pass_states_[idx];

    if (bw_path_state.velocity >
        path_velocity_limit.calculateOverallLimit(joint_path_derivatives)) {
      std::cout << "Error: bw path state velocity above MVC at idx " << idx
                << std::endl;
      return Result::GENERAL_ERROR;
    }
  }

  for (size_t idx = 0; idx < trajectory.size(); ++idx) {
    const auto &state = trajectory[idx];
    const auto fw_path_state = state.path_state;
    const auto &prev_bw_state = backward_pass_states_[state.bw_idx];
    const auto &next_bw_state = backward_pass_states_[state.bw_idx + 1];
    const auto joint_state = state.waypoint.joints;

    if (!(prev_bw_state.position < fw_path_state.position &&
          next_bw_state.position > fw_path_state.position)) {
      std::cout << "Error: bw path states idx not located before and after fw "
                   "state position at idx "
                << idx << std::endl;
      return Result::GENERAL_ERROR;
    }

    if (fw_path_state.velocity > prev_bw_state.velocity &&
        fw_path_state.velocity > next_bw_state.velocity) {
      std::cout << "Error: fw path state velocity above bw path state velocity "
                   "at idx "
                << idx << ", bw idx " << state.bw_idx << std::endl;
      return Result::GENERAL_ERROR;
    }

    // limits are computed at the next time instant for the current position
    const auto acc_max = trajectory[idx + 1].path_state.acc_max;
    const auto acc_min = trajectory[idx + 1].path_state.acc_min;
    if (fw_path_state.acceleration > acc_max + cycle_time_ ||
        fw_path_state.acceleration < acc_min - cycle_time_) {
      std::cout << "Error: fw path acceleration not within limits at idx "
                << idx << " and position " << fw_path_state.position
                << std::endl;
      std::cout << "limits: " << fw_path_state.acc_max << ", "
                << fw_path_state.acc_min << "; path acceleration "
                << fw_path_state.acceleration << std::endl;
      return Result::GENERAL_ERROR;
    }

    if (idx > 0) {
      const auto prev_fw_path_state = trajectory[idx - 1].path_state;
      const auto vel_time_diff =
          (fw_path_state.position - prev_fw_path_state.position) / cycle_time_;
      if (vel_time_diff - cycle_time_ > fw_path_state.velocity) {
        std::cout << "Error: time difference of path position greater than "
                     "path velocity at idx "
                  << idx << ", time diff " << vel_time_diff << ", path vel "
                  << fw_path_state.velocity << std::endl;
        return Result::GENERAL_ERROR;
      }
      const auto accel_time_diff =
          (vel_time_diff - prev_fw_path_state.velocity) / cycle_time_;
      if (accel_time_diff - cycle_time_ > fw_path_state.acceleration) {
        std::cout << "Error: time difference of path velocity greater than "
                     "path acceleration at idx "
                  << idx << std::endl;
        return Result::GENERAL_ERROR;
      }
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
              constraints_.joints.acceleration_max[joint] + cycle_time_ ||
          joint_state.acceleration[joint] <
              constraints_.joints.acceleration_min[joint] - cycle_time_) {
        std::cout << "Error: joint acceleration not within limits at idx "
                  << idx << " and joint " << joint << std::endl;
        std::cout << "limits: " << constraints_.joints.acceleration_max[joint]
                  << ", " << constraints_.joints.acceleration_min[joint]
                  << "; joint acceleration " << joint_state.acceleration[joint]
                  << std::endl;
        return Result::GENERAL_ERROR;
      }
    }
  }

  // TODO(wolfgang): also take the time difference of the joint position signal
  // and check if its within limits

  double max_normalized_vel = 0.0, max_normalized_accel = 0.0;
  std::vector<WaypointJoint<N_JOINTS>> joint_trajectory_normalized(
      trajectory.size());
  Eigen::VectorXd infinity_norm(num_idx_);
  for (size_t idx = 0; idx < trajectory.size(); ++idx) {
    const auto &joint_state = trajectory[idx].waypoint.joints;
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

// TODO(wolfgang): refactor with the duplicate code above
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
    if (fw_path_state.acceleration >
            fw_path_state.acc_max + ACCELERATION_TOLERANCE ||
        fw_path_state.acceleration <
            fw_path_state.acc_min - ACCELERATION_TOLERANCE) {
      std::cout << "Error: fw path acceleration not within limits at idx "
                << idx << std::endl;
      std::cout << "limits: " << fw_path_state.acc_max << ", "
                << fw_path_state.acc_min << "; path acceleration "
                << fw_path_state.acceleration << "; diff: "
                << fw_path_state.acc_min - fw_path_state.acceleration
                << std::endl;
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
              constraints_.joints.acceleration_max[joint] +
                  ACCELERATION_TOLERANCE ||
          joint_state.acceleration[joint] <
              constraints_.joints.acceleration_min[joint] -
                  ACCELERATION_TOLERANCE) {
        std::cout << "Error: joint acceleration not within limits at idx "
                  << idx << " and joint " << joint << std::endl;
        std::cout << "limits: " << constraints_.joints.acceleration_max[joint]
                  << ", " << constraints_.joints.acceleration_min[joint]
                  << "; joint acceleration " << joint_state.acceleration[joint]
                  << std::endl;
        return Result::GENERAL_ERROR;
      }
    }
  }

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
nlohmann::json RTTOPP2<N_JOINTS, MAX_WAYPOINTS, MAX_STEPS>::toJson(
    const Waypoints<N_JOINTS> &waypoints,
    const Trajectory<N_JOINTS> &trajectory) const {
  nlohmann::json j;
  PathVelocityLimit<N_JOINTS> path_velocity_limit(constraints_.joints);

  for (size_t idx = 0; idx < waypoints.size(); ++idx) {
    const auto &waypoint = waypoints[idx];
    j["waypoints"][idx]["idx"] = idx;
    j["waypoints"][idx]["idx_on_path"] =
        preprocess_.getWaypointPathIndices(idx);
    std::vector<double> joint_positions, joint_velocities;
    utils::setMatrixAsVector(joint_positions, waypoint.joints.position);
    utils::setMatrixAsVector(joint_velocities, waypoint.joints.velocity);
    j["waypoints"][idx]["joints"]["position"] = joint_positions;
    j["waypoints"][idx]["joints"]["velocity"] = joint_velocities;
  }

  for (size_t idx = 0; idx < num_idx_; ++idx) {
    j["path_parameterization_bw"][idx]["idx"] = idx;

    const auto &backward_state = backward_pass_states_[idx];

    j["path_parameterization_bw"][idx]["position"] = backward_state.position;
    j["path_parameterization_bw"][idx]["velocity"] = backward_state.velocity;
    j["path_parameterization_bw"][idx]["acceleration"] =
        backward_state.acceleration;
    j["path_parameterization_bw"][idx]["acc_min"] = backward_state.acc_min;
    j["path_parameterization_bw"][idx]["acc_max"] = backward_state.acc_max;

    // data, that we need for phase space plotting and that is not directly
    // computed here: absolute MVC (first-order + second-order)
    const auto joint_path_derivatives = preprocess_.getDerivatives(idx);
    // also save derivatives for debugging
    std::vector<double> derivatives_first, derivatives_second,
        derivatives_third;
    utils::setMatrixAsVector(derivatives_first, joint_path_derivatives.first);
    utils::setMatrixAsVector(derivatives_second, joint_path_derivatives.second);
    utils::setMatrixAsVector(derivatives_third, joint_path_derivatives.third);
    j["path_parameterization_bw"][idx]["derivatives"]["first"] =
        derivatives_first;
    j["path_parameterization_bw"][idx]["derivatives"]["second"] =
        derivatives_second;
    j["path_parameterization_bw"][idx]["derivatives"]["third"] =
        derivatives_third;
    j["path_parameterization_bw"][idx]["vel_abs_max"] =
        path_velocity_limit.calculateOverallLimit(joint_path_derivatives);
    j["path_parameterization_bw"][idx]["vel_abs_max_first"] =
        path_velocity_limit.calculateJointVelocityLimit(joint_path_derivatives);
    j["path_parameterization_bw"][idx]["vel_abs_max_second"] =
        path_velocity_limit.calculateJointAccelerationLimit(
            joint_path_derivatives);

    // TODO(wolfgang): set same joint constraints for every idx. When
    // considering dynamics, replace these with computed constraints
    std::vector<double> joint_acceleration_min;
    utils::setMatrixAsVector(joint_acceleration_min,
                             constraints_.joints.acceleration_min);
    j["path_parameterization_bw"][idx]["joint_constraints"]["acc_min"] =
        joint_acceleration_min;

    std::vector<double> joint_acceleration_max;
    utils::setMatrixAsVector(joint_acceleration_max,
                             constraints_.joints.acceleration_max);
    j["path_parameterization_bw"][idx]["joint_constraints"]["acc_max"] =
        joint_acceleration_max;
  }

  for (size_t idx = 0; idx < trajectory.size(); ++idx) {
    j["path_parameterization_fw"][idx]["idx"] = idx;

    const auto &state = trajectory[idx];
    const auto fw_path_state = state.path_state;
    const auto joint_state = state.waypoint.joints;

    j["path_parameterization_fw"][idx]["position"] = fw_path_state.position;
    j["path_parameterization_fw"][idx]["velocity"] = fw_path_state.velocity;
    j["path_parameterization_fw"][idx]["acceleration"] =
        fw_path_state.acceleration;
    j["path_parameterization_fw"][idx]["acc_min"] = fw_path_state.acc_min;
    j["path_parameterization_fw"][idx]["acc_max"] = fw_path_state.acc_max;
    j["path_parameterization_fw"][idx]["time"] = state.time;
    j["path_parameterization_fw"][idx]["bw_idx"] = state.bw_idx;

    // data, that we need for phase space plotting and that is not directly
    // computed here: absolute MVC (first-order + second-order)
    const auto joint_path_derivatives = state.derivatives;
    // also save derivatives for debugging
    std::vector<double> derivatives_first, derivatives_second,
        derivatives_third;
    utils::setMatrixAsVector(derivatives_first, joint_path_derivatives.first);
    utils::setMatrixAsVector(derivatives_second, joint_path_derivatives.second);
    utils::setMatrixAsVector(derivatives_third, joint_path_derivatives.third);
    j["path_parameterization_fw"][idx]["derivatives"]["first"] =
        derivatives_first;
    j["path_parameterization_fw"][idx]["derivatives"]["second"] =
        derivatives_second;
    j["path_parameterization_fw"][idx]["derivatives"]["third"] =
        derivatives_third;
    j["path_parameterization_fw"][idx]["vel_abs_max"] =
        path_velocity_limit.calculateOverallLimit(joint_path_derivatives);
    j["path_parameterization_fw"][idx]["vel_abs_max_first"] =
        path_velocity_limit.calculateJointVelocityLimit(joint_path_derivatives);
    j["path_parameterization_fw"][idx]["vel_abs_max_second"] =
        path_velocity_limit.calculateJointAccelerationLimit(
            joint_path_derivatives);

    j["path_parameterization_fw"][idx]["jointtrajpoints"]["idx"] = idx;
    j["path_parameterization_fw"][idx]["jointtrajpoints"]["values"] =
        utils::jointStateToJson(joint_state);
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

// TODO(wolfgang): refactor this with duplicate code above
template <size_t N_JOINTS, size_t MAX_WAYPOINTS, size_t MAX_STEPS>
nlohmann::json RTTOPP2<N_JOINTS, MAX_WAYPOINTS, MAX_STEPS>::toJson(
    const Waypoints<N_JOINTS> &waypoints) const {
  nlohmann::json j;
  PathVelocityLimit<N_JOINTS> path_velocity_limit(constraints_.joints);

  for (size_t idx = 0; idx < waypoints.size(); ++idx) {
    const auto &waypoint = waypoints[idx];
    j["waypoints"][idx]["idx"] = idx;
    j["waypoints"][idx]["idx_on_path"] =
        preprocess_.getWaypointPathIndices(idx);
    std::vector<double> joint_positions, joint_velocities;
    utils::setMatrixAsVector(joint_positions, waypoint.joints.position);
    utils::setMatrixAsVector(joint_velocities, waypoint.joints.velocity);
    j["waypoints"][idx]["joints"]["position"] = joint_positions;
    j["waypoints"][idx]["joints"]["velocity"] = joint_velocities;
  }

  for (size_t idx = 0; idx < num_idx_; ++idx) {
    j["path_parameterization"][idx]["idx"] = idx;

    const auto &forward_state = forward_pass_states_[idx];
    const auto &backward_state = backward_pass_states_[idx];

    j["path_parameterization"][idx]["position"] = backward_state.position;
    j["path_parameterization"][idx]["forward"]["velocity"] =
        forward_state.velocity;
    j["path_parameterization"][idx]["forward"]["acceleration"] =
        forward_state.acceleration;
    j["path_parameterization"][idx]["acc_min"] = forward_state.acc_min;
    j["path_parameterization"][idx]["acc_max"] = forward_state.acc_max;

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
