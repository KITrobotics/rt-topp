#pragma once

#include <limits>
#include <vector>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

namespace rttopp {

class Result {
 public:
  enum Msg : uint8_t {
    SUCCESS = 0,

    // errors
    ERRORS_BEGIN = 64,
    GENERAL_ERROR = ERRORS_BEGIN,
    // TODO(wolfgang): set here all (specific) errors that can occur and add
    // warnings if useful
    INVALID_INPUT,

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

  [[nodiscard]] bool set() const { return value_ != NOT_SET; }

  [[nodiscard]] const char *message() const;

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

// TODO(wolfgang): the data types below are (likely) only for internal use, so
// consider moving them to their respective classes and don't expose them here
// to the global project scope

struct PathState {
  double position;
  double velocity = 0.0;
  double acceleration = 0.0;
  double jerk = 0.0;

  // dynamic max and min acceleration
  double acc_max, acc_min = 0.0;
  // velocity is on the second-order MVC
  bool on_mvc_second = false;
  // velocity is on the limit curve given by the first pass
  bool on_previous_limit_curve = false;
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

namespace utils {

// TODO(wolfgang): kinematics uses -07, is this value ok here?
constexpr double EPS = 1.0e-06;

[[nodiscard]] constexpr bool isZero(double v) { return std::abs(v) < EPS; }

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

  return j;
}
}  // namespace utils
}  // namespace rttopp
