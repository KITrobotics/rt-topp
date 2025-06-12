#pragma once

/* #include <iostream> */

#include <limits>
#include <optional>
#include <vector>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

namespace rttopp {

class Result {
 public:
  enum Msg : uint8_t {
    SUCCESS = 0,
    WORKING,   // sample() and keepParam(), goal state not yet reached
    FINISHED,  // sample() and keepParam(), goal state reached

    // errors
    ERRORS_BEGIN = 64,
    // TODO(wolfgang): set here all (specific) errors that can occur and add
    // warnings if useful
    START_STATE_VELOCITY_TOO_HIGH = ERRORS_BEGIN,
    GOAL_STATE_VELOCITY_TOO_HIGH,
    NOT_SOLVABLE_VELOCITY_STALLING,
    INVALID_INPUT,
    GENERAL_ERROR,

    NOT_SET = 255
  };

  Result() = default;
  Result(Msg msg)  // NOLINT google-explicit-constructor
      : value_(msg) {}

  [[nodiscard]] bool error() const { return value_ >= ERRORS_BEGIN; }

  [[nodiscard]] bool success() const {
    // yellow means green, like with traffic lights :)
    return value_ < ERRORS_BEGIN;
  }

  [[nodiscard]] Msg val() const { return value_; }

  [[nodiscard]] bool set() const { return value_ != NOT_SET; }

  [[nodiscard]] const char *message() const {
    switch (value_) {
      case SUCCESS:
        return "SUCCESS";
      case START_STATE_VELOCITY_TOO_HIGH:
        return "START_STATE_VELOCITY_TOO_HIGH";
      case GOAL_STATE_VELOCITY_TOO_HIGH:
        return "GOAL_STATE_VELOCITY_TOO_HIGH";
      case NOT_SOLVABLE_VELOCITY_STALLING:
        return "NOT_SOLVABLE_VELOCITY_STALLING";
      case INVALID_INPUT:
        return "INVALID_INPUT";
      case GENERAL_ERROR:
        return "GENERAL_ERROR";
      case NOT_SET:
        return "NOT_SET";

      default:
        return "UNKNOWN STATUS CODE";
    }
  }

 private:
  Msg value_{NOT_SET};
};

template <size_t N_JOINTS>
struct JointConstraints {
  // TODO(wolfgang): only Eigen 3.4 has support for initializer lists, add this
  // version as submodule? Support for initializer lists would help to make
  // Eigen Matrix behave like a regular array
  using JointConstraintType = Eigen::Matrix<double, N_JOINTS, 1>;

  JointConstraintType velocity_max =
      std::numeric_limits<double>::max() * JointConstraintType::Ones();
  // TODO(wolfgang): use std::optional for min, jerk and torque constraints?
  JointConstraintType velocity_min =
      std::numeric_limits<double>::lowest() * JointConstraintType::Ones();
  JointConstraintType acceleration_max =
      std::numeric_limits<double>::max() * JointConstraintType::Ones();
  JointConstraintType acceleration_min =
      std::numeric_limits<double>::lowest() * JointConstraintType::Ones();
  JointConstraintType jerk_max =
      std::numeric_limits<double>::max() * JointConstraintType::Ones();
  JointConstraintType jerk_min =
      std::numeric_limits<double>::lowest() * JointConstraintType::Ones();
  JointConstraintType torque_max =
      std::numeric_limits<double>::max() * JointConstraintType::Ones();
  JointConstraintType torque_min =
      std::numeric_limits<double>::lowest() * JointConstraintType::Ones();
};

struct CartesianConstraints {
  // TODO(wolfgang): remove the ones that we don't support
  struct CartesianConstraintGroup {
    double velocity_max = std::numeric_limits<double>::max();
    std::optional<double> acceleration_max,
        jerk_max = std::numeric_limits<double>::max();
  };

  CartesianConstraintGroup tra;
  std::optional<CartesianConstraintGroup> rot;
};

template <size_t N_JOINTS>
struct Constraints {
  JointConstraints<N_JOINTS> joints;
  std::optional<CartesianConstraints> cart;
};

template <size_t N_DOF>
using WaypointJointDataType = Eigen::Matrix<double, N_DOF, 1>;

template <size_t N_DOF>
struct WaypointJoint {
  WaypointJointDataType<N_DOF> position;
  WaypointJointDataType<N_DOF> velocity = WaypointJointDataType<N_DOF>::Zero();
  WaypointJointDataType<N_DOF> acceleration =
      WaypointJointDataType<N_DOF>::Zero();

  double time = 0.0;
};

struct WaypointCartesian {
  WaypointJoint<3> tra;
  WaypointJoint<4> rot;
};

template <size_t N_JOINTS>
struct Waypoint {
  WaypointJoint<N_JOINTS> joints;
  WaypointCartesian cart;
};

// TODO(wolfgang): is this ok, not using static vector with max number for input
// waypoints? max num would need to be specified separately for the rttopp
// instance anyway and users need to reserve here themselves if they need
// real-time
template <size_t N_JOINTS>
using Waypoints = std::vector<Waypoint<N_JOINTS>>;

struct PathState {
  double position;
  double velocity = 0.0;
  double acceleration = 0.0;
  /* double jerk = 0.0; */

  // dynamic max and min acceleration
  double acc_max, acc_min = 0.0;
};

template <std::size_t N_JOINTS>
struct JointPathDerivatives {
  Eigen::Matrix<double, N_JOINTS, 1> first =
      Eigen::Matrix<double, N_JOINTS, 1>::Ones();
  Eigen::Matrix<double, N_JOINTS, 1> second =
      Eigen::Matrix<double, N_JOINTS, 1>::Zero();
  Eigen::Matrix<double, N_JOINTS, 1> third =
      Eigen::Matrix<double, N_JOINTS, 1>::Zero();
};

template <size_t N_JOINTS>
struct State {
  Waypoint<N_JOINTS> waypoint;
  double time;

  PathState path_state;
  JointPathDerivatives<N_JOINTS> derivatives;
  size_t
      bw_idx;  // idx from bw pass, position directly before path_state position
};

// full trajectory when sampling whole path at once
// not real-time
template <size_t N_JOINTS>
using Trajectory = std::vector<State<N_JOINTS>>;

namespace utils {

constexpr double EPS = 1.0e-07;

[[nodiscard]] constexpr bool isZero(double v) { return std::abs(v) <= EPS; }

/**
 * @brief Just a constexpr replacement for std::pow when y is int for
 * substantially faster evaluation
 *
 * @tparam T
 * @param x
 * @param y
 *
 * @return
 */
template <typename T>
constexpr T pow(T x, unsigned int y) {
  return y == 0 ? 1.0 : x * pow(x, y - 1);
}

// TODO(wolfgang): remove this if calculateTimeDiff() proves to be the better
// solution
/* [[nodiscard]] static double getTimeStampDiff(const PathState &current_state,
 */
/*                                 double next_state_pos) { */

/*   /\* solve with vel at next pos, otherwise may not be stable!! *\/ */

/*   // p = p0 + v0*t + 0.5 * a * t^2 */
/*   // -> a * x^2 + b * x + c = 0 */
/*   const double a = current_state.acceleration / 2.0; */
/*   const double b = current_state.velocity; */
/*   double c = current_state.position - next_state_pos; */

/*   if (isZero(a)) { */
/*     return -c / b; */
/*   } */

/*   /\* if (isZero(b)) { *\/ */
/*   /\*   c = -c; *\/ */
/*   /\* } *\/ */

/*   double disc = utils::pow(b, 2) - 4.0 * a * c; */
/*   double den = 2.0 * a; */
/*   double p1 = -b / den; */
/*   double p2 = std::sqrt(disc) / den; */
/*   double s1 = p1 + p2; */
/*   double s2 = p1 -p2; */

/*   if (s1 > 0.0 && s2 > 0.0) { */
/*     return std::min(s1, s2); */
/*   } */

/*   if (s1 < 0.0 && s2 < 0.0) { */
/*     std::cout << "both time sol < 0!!" << std::endl; */
/*   } */

/*   if (disc < 0) { */
/*     std::cout << "disc < 0!!, b " << b << ", a " << a << ", c " << c <<
 * std::endl; */
/*     // TODO(wolfgang): really the correct solution? */
/*     return 0.0; */
/*   } */

/*   if (!std::isfinite(s1) || !std::isfinite(s2)) { */
/*     std::cout << "one value nan!!" << std::endl; */
/*   } */

/*   return s1 > 0.0 ? s1 : s2; */
/* } */

/**
 * @brief Creates a vector container e.g std::vector from Eigen matrix
 *
 * @tparam ScalarType
 * @tparam VectorType
 * @tparam Rows
 * @tparam Cols
 * @param v
 * @param mat
 */
template <typename ScalarType = double, typename VectorType, int Rows, int Cols>
void setMatrixAsVector(VectorType &v,
                       const Eigen::Matrix<ScalarType, Rows, Cols> &mat) {
  v.resize(mat.rows() * mat.cols());
  Eigen::Map<Eigen::Matrix<ScalarType, Rows, Cols>>(
      reinterpret_cast<ScalarType *>(v.data()), mat.rows(), mat.cols()) = mat;
}

template <size_t N_JOINTS>
nlohmann::json jointStateToJson(const WaypointJoint<N_JOINTS> &joint_state) {
  nlohmann::json j;

  const auto eigen_to_std = [](const Eigen::MatrixXd &mat) {
    return std::vector<double>(mat.data(),
                               mat.data() + mat.rows() * mat.cols());
  };

  j["angle"] = eigen_to_std(joint_state.position);
  j["velocity"] = eigen_to_std(joint_state.velocity);
  j["acceleration"] = eigen_to_std(joint_state.acceleration);
  j["time"] = joint_state.time;

  return j;
}
}  // namespace utils
}  // namespace rttopp
