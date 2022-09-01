#pragma once

#include <boost/container/static_vector.hpp>
#include <fstream>
#include <rttopp/types_utils.h>

namespace rttopp {

// TODO(wolfgang): mirror MAX_WAYPOINTS and MAX_STEPS defaults from rttopp2,
// maybe create common consts in types_utils for that? (different again if we
// switch to std::vectors and don't use Boost's static_vector)
template <size_t N_JOINTS, size_t N_MAX_WAYPOINTS = 30,
          size_t N_MAX_STEPS = 6000>
class Preprocessing {
 public:
  explicit Preprocessing(const double path_resolution = 0.05,
                         const size_t steps_per_cycle = 0)
      : path_resolution_(path_resolution), steps_per_cycle_(steps_per_cycle) {}

  ~Preprocessing() = default;

  using NJointWaypointDataType = WaypointJointDataType<N_JOINTS>;
  using DoubleCoeffContainer =
      boost::container::static_vector<double, N_MAX_WAYPOINTS>;
  using MatrixCoeffContainer =
      boost::container::static_vector<NJointWaypointDataType, N_MAX_WAYPOINTS>;
  using DerivativesDataType = JointPathDerivatives<N_JOINTS>;
  /**
   * @brief Process the waypoints such that the intepolants and their partial
   * derivatives are computed in a resolution of `s_res`.
   *
   * @param wps Input waypoints
   * @param s_res Resolution of the interpolation
   */
  // TODO(wolfgang): needs make_pair result to at least give warning that
  // N_MAX_STEPS is reached, and that N_MAX_WAYPOINTS is exceeded and not all
  // waypoints are used
  [[nodiscard]] size_t processWaypoints(const Waypoints<N_JOINTS>& wps);

  [[nodiscard]] JointPathDerivatives<N_JOINTS> getDerivatives(
      size_t idx) const {
    assert(idx < n_seg_);
    return derivatives_[idx];
  };

  /**
   * @brief Compute the joint position and partial derivatives given a position
   * along the path `s`.
   *
   * @param s_pos
   * @return std::pair<NJointWaypointDataType, JointPathDerivatives<N_JOINTS>>
   */
  // TODO(Xi): should method be const?
  [[nodiscard]] std::pair<NJointWaypointDataType,
                          JointPathDerivatives<N_JOINTS>>
  computeJointPositionAndDerivatives(double s_pos);

  /**
   * @brief Get the position of the variable s and the joint position
   * given the index of s.
   *
   * @param s_idx The index of the path variable
   * @return double The returned position of s at this index
   */
  [[nodiscard]] double getPathPosition(size_t s_idx) const;

  /**
   * @brief Get the Joint Position From Path object
   *
   * @param idx
   * @return NJointWaypointDataType
   */
  [[nodiscard]] NJointWaypointDataType getJointPositionFromPath(
      const size_t idx) const {
    assert(idx < n_seg_);
    return interpl_[idx].joints.position;
  };

  /**
   * @brief Get the number of interpolated segments
   *
   * @return size_t
   */
  [[nodiscard]] size_t getSegmentSize() const { return interpl_.size(); }

  [[nodiscard]] size_t getDerivativesSize() const {
    return derivatives_.size();
  }

  void outputDataAsCSV(const std::string& file_path = "/tmp/spline.csv");

  /**
   * @brief Get the interpolation ratio from path object
   *
   * @param s_idx
   * @return double
   */
  [[nodiscard]] double getTauFromPath(size_t s_idx) const {
    return tau_[s_idx];
  }

  /**
   * @brief Return the index of the waypoint on the final paramterized path
   *
   * @param wps_idx Index of the input waypoint
   * @return size_t Index of this waypoint on the final path
   */
  [[nodiscard]] size_t getWaypointPathIndices(size_t wps_idx) const {
    return wp_s_idx_[wps_idx];
  }

 private:
  /**
   * @brief Compute the path variable using the L2-norm between waypoints.
   *
   * @param wps Input waypoints
   */
  template <typename PathContainer>
  void computePathProjectionBasic(const Waypoints<N_JOINTS>& wps,
                                  PathContainer& s);

  /**
   * @brief Compute the spline coefficients given the waypoints and the path
   * varibale. The path varible is the reference coordinates.
   *
   * @param wps Input waypoints
   */
  template <typename PathContainer>
  void computeCoefficients(const Waypoints<N_JOINTS>& wps,
                           PathContainer& s_wps);

  /**
   * @brief Compute the spline coefficients given the waypoints. Update the
   * path variable in a iterative manner.
   *
   * @param wps Input waypoints
   */
  template <typename PathContainer>
  void computeCoefficientsIterarive(const Waypoints<N_JOINTS>& wps,
                                    PathContainer& s_wps);

  /**
   * @brief
   *
   * @param s
   * @return DerivativesDataType
   */
  [[nodiscard]] DerivativesDataType computeDerivatives(double s);

  /**
   * @brief
   *
   */
  void computeDerivativesFull();

  /**
   * @brief Prepare the entries of the matrix to compute spline coefficients
   *
   * @param wps Input waypoints
   */
  template <typename PathContainer>
  void prepareCoeffMatrix(const Waypoints<N_JOINTS>& wps, PathContainer& s_wps);

  /**
   * @brief
   *
   * @tparam PathContainer
   * @param wps
   * @param s
   * @param res
   * @param keep_wps
   */
  template <typename PathContainer, typename PathContainer2>
  Waypoints<N_JOINTS> interpolateBasic(const Waypoints<N_JOINTS>& wps,
                                       PathContainer& s_wps,
                                       PathContainer2& s_adjusted, double res);

  /**
   * @brief
   *
   * @tparam PathContainer
   * @param wps
   * @param s
   * @param res
   * @param keep_wps
   */
  template <typename PathContainer, typename PathContainer2>
  void interpolateFinal(const Waypoints<N_JOINTS>& wps, PathContainer& s_wps,
                        PathContainer2& s_adjusted, double res,
                        bool keep_wps = true);

  boost::container::static_vector<Waypoint<N_JOINTS>, N_MAX_STEPS> interpl_;

  MatrixCoeffContainer val_d_;
  MatrixCoeffContainer coeff_k_;
  boost::container::static_vector<DerivativesDataType, N_MAX_STEPS>
      derivatives_;

  size_t n_coeff_;
  size_t n_seg_;

  const double path_resolution_;
  const double steps_per_cycle_;

  boost::container::static_vector<size_t, N_MAX_WAYPOINTS> wp_s_idx_;
  DoubleCoeffContainer val_a_;
  DoubleCoeffContainer val_b_;
  DoubleCoeffContainer val_c_;

  boost::container::static_vector<double, N_MAX_WAYPOINTS> s_wps_;
  boost::container::static_vector<double, N_MAX_STEPS> s_;
  boost::container::static_vector<double, N_MAX_STEPS> s_intermediate_;
  boost::container::static_vector<double, N_MAX_STEPS> tau_;

};  // End of the class declaration.

template <size_t N_JOINTS, size_t N_MAX_WAYPOINTS, size_t N_MAX_STEPS>
size_t Preprocessing<N_JOINTS, N_MAX_WAYPOINTS, N_MAX_STEPS>::processWaypoints(
    const Waypoints<N_JOINTS>& wps) {
  auto n_input_wps = wps.size();
  n_coeff_ = n_input_wps > N_MAX_WAYPOINTS ? N_MAX_WAYPOINTS : n_input_wps;

  tau_.clear();
  s_wps_.clear();
  s_intermediate_.clear();
  wp_s_idx_.clear();

  // Compute the coefficients
  computePathProjectionBasic(wps, s_wps_);

  // TODO(wolfgang): resize is O(N), can't this have this size directly in the
  // declaration or use an array?
  // coeff_k_.resize(N_MAX_WAYPOINTS);
  computeCoefficients(wps, s_wps_);

  // Interpolate for the first time
  auto interpolated_wps =
      interpolateBasic(wps, s_wps_, s_intermediate_, path_resolution_);

  // Compute the coefficent with adjusted path vairable s
  computeCoefficients(interpolated_wps, s_intermediate_);

  // Interpolate for the second time with
  interpolateFinal(interpolated_wps, s_intermediate_, s_, path_resolution_);

  return n_seg_;
}

template <size_t N_JOINTS, size_t N_MAX_WAYPOINTS, size_t N_MAX_STEPS>
std::pair<WaypointJointDataType<N_JOINTS>, JointPathDerivatives<N_JOINTS>>
Preprocessing<N_JOINTS, N_MAX_WAYPOINTS,
              N_MAX_STEPS>::computeJointPositionAndDerivatives(double s_pos) {
  // This implementation offer random access given a path position s_pos.
  // We can optimize the performance if the derivatives is computed in order

  // Step 1: Get the two waypoints related to this position.
  size_t start_wp = 0;
  size_t end_wp = n_coeff_ - 1;

  // If the start and end waypoint is not next to each other
  // Avoid using subtraction for size_t
  while (start_wp + 1 < end_wp) {
    size_t tmp_wp = (start_wp + end_wp) / 2;
    double tmp_s = s_[wp_s_idx_[tmp_wp]];
    end_wp = tmp_s >= s_pos ? tmp_wp : end_wp;
    start_wp = tmp_s >= s_pos ? start_wp : tmp_wp;
  }
  assert(end_wp > start_wp);

  // Step 2: Get the segment related to this position.
  size_t start_seg = wp_s_idx_[start_wp];
  size_t end_seg = wp_s_idx_[end_wp];
  while (start_seg + 1 < end_seg) {
    size_t tmp_seg = (start_seg + end_seg) / 2;
    double tmp_s = s_[tmp_seg];
    end_seg = tmp_s >= s_pos ? tmp_seg : end_seg;
    start_seg = tmp_s >= s_pos ? start_wp : tmp_seg;
  }
  assert(end_seg > start_seg);

  // Step 3: Compute the tau using linear interpolation
  double start_s_segment = s_[start_seg];
  double end_s_segment = s_[end_seg];
  double d_s_segments = end_s_segment - start_s_segment;

  double start_tau = (start_seg == wp_s_idx_[start_wp]) ? .0 : tau_[start_seg];
  double end_tau = (end_seg == wp_s_idx_[end_wp]) ? 1.0 : tau_[end_seg];
  double tau =
      (s_pos - start_s_segment) / d_s_segments * (end_tau - start_tau) +
      start_tau;
  assert(tau > start_tau);
  assert(tau < end_tau);

  NJointWaypointDataType start_position_wp =
      interpl_[wp_s_idx_[start_wp]].joints.position;
  NJointWaypointDataType d_pos_wp =
      interpl_[wp_s_idx_[end_wp]].joints.position - start_position_wp;

  // We have to use the original coordinate, otherwise the coefficients are not
  // correct
  double d_s_wp = s_intermediate_[end_wp] - s_intermediate_[start_wp];
  NJointWaypointDataType a_i = coeff_k_[start_wp] * d_s_wp - d_pos_wp;
  NJointWaypointDataType b_i = -coeff_k_[end_wp] * d_s_wp + d_pos_wp;

  DerivativesDataType der;
  double tau2 = tau * tau;
  double tau3 = tau2 * tau;
  NJointWaypointDataType joint_position =
      start_position_wp + tau * (a_i + d_pos_wp) + tau2 * (b_i - 2 * a_i) +
      (a_i - b_i) * tau3;
  der.first =
      (a_i + d_pos_wp + 2 * (b_i - 2 * a_i) * tau + 3 * (a_i - b_i) * tau2) /
      d_s_wp;
  der.second = (2 * (b_i - 2 * a_i) + 6 * (a_i - b_i) * tau) / d_s_wp / d_s_wp;
  der.third = 6 * (a_i - b_i) / d_s_wp / d_s_wp / d_s_wp;

  return {joint_position, der};
}

template <size_t N_JOINTS, size_t N_MAX_WAYPOINTS, size_t N_MAX_STEPS>
double Preprocessing<N_JOINTS, N_MAX_WAYPOINTS, N_MAX_STEPS>::getPathPosition(
    size_t s_idx) const {
  return s_[s_idx];
}

template <size_t N_JOINTS, size_t N_MAX_WAYPOINTS, size_t N_MAX_STEPS>
void Preprocessing<N_JOINTS, N_MAX_WAYPOINTS, N_MAX_STEPS>::outputDataAsCSV(
    const std::string& file_path) {
  std::ofstream myfile;
  myfile.open(file_path.c_str());
  const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision,
                                         Eigen::DontAlignCols, ", ", "\n");

  for (std::size_t idx = 0; idx < n_seg_; ++idx) {
    auto our_derivatives = this->getDerivatives(idx);
    auto s_position = this->getPathPosition(idx);
    rttopp::WaypointJointDataType<N_JOINTS> our_position =
        this->getJointPositionFromPath(idx);
    myfile << s_position << ", " << our_position.transpose().format(CSVFormat)
           << ", " << our_derivatives.first.transpose().format(CSVFormat) << ","
           << our_derivatives.second.transpose().format(CSVFormat) << ","
           << std::endl;
  }
  myfile.close();
}

template <size_t N_JOINTS, size_t N_MAX_WAYPOINTS, size_t N_MAX_STEPS>
template <typename PathContainer>
void Preprocessing<N_JOINTS, N_MAX_WAYPOINTS, N_MAX_STEPS>::
    computePathProjectionBasic(const Waypoints<N_JOINTS>& wps,
                               PathContainer& s_wps) {
  s_wps.clear();
  s_wps.emplace_back(.0);

  for (size_t i = 1; i < n_coeff_; ++i) {
    auto last_wp = wps[i - 1];
    auto this_wp = wps[i];
    double diff_s = (this_wp.joints.position - last_wp.joints.position).norm();
    double this_s = diff_s + s_wps[i - 1];
    s_wps.emplace_back(this_s);
  }
}

template <size_t N_JOINTS, size_t N_MAX_WAYPOINTS, size_t N_MAX_STEPS>
JointPathDerivatives<N_JOINTS>
Preprocessing<N_JOINTS, N_MAX_WAYPOINTS, N_MAX_STEPS>::computeDerivatives(
    double s) {
  DerivativesDataType der;

  auto s_minus_delta = s + rttopp::utils::EPS;
  auto joints_and_derivatives_plus_delta =
      computeJointPositionAndDerivatives(s_minus_delta);
  auto interpolated_joints_minus_delta =
      joints_and_derivatives_plus_delta.first;

  auto joints_and_derivatives = computeJointPositionAndDerivatives(s);
  auto interpolated_joints = joints_and_derivatives.first;

  auto derivatives_plus_delta = joints_and_derivatives_plus_delta.second;
  auto derivatives = joints_and_derivatives.second;

  der.first = (interpolated_joints_minus_delta - interpolated_joints) /
              rttopp::utils::EPS;

  double nf = der.first.norm() / derivatives.first.norm();
  der.second = (derivatives_plus_delta.first - derivatives.first) /
               rttopp::utils::EPS * nf;
  der.third = (derivatives_plus_delta.second - derivatives.second) /
              rttopp::utils::EPS * nf;
  return der;
}

template <size_t N_JOINTS, size_t N_MAX_WAYPOINTS, size_t N_MAX_STEPS>
void Preprocessing<N_JOINTS, N_MAX_WAYPOINTS,
                   N_MAX_STEPS>::computeDerivativesFull() {
  derivatives_.clear();
  for (size_t seg = 0; seg < n_seg_; ++seg) {
    double segment_s = s_[seg];
    DerivativesDataType der = computeDerivatives(segment_s);
    derivatives_.emplace_back(der);
  }
}

template <size_t N_JOINTS, size_t N_MAX_WAYPOINTS, size_t N_MAX_STEPS>
template <typename PathContainer>
void Preprocessing<N_JOINTS, N_MAX_WAYPOINTS, N_MAX_STEPS>::computeCoefficients(
    const Waypoints<N_JOINTS>& wps, PathContainer& s_wps) {
  prepareCoeffMatrix(wps, s_wps);
  size_t n_wps = wps.size();

  NJointWaypointDataType desired_start_vel = wps[0].joints.velocity;
  NJointWaypointDataType desired_end_vel = wps[n_wps - 1].joints.velocity;

  bool start_is_natural = utils::isZero(desired_start_vel.norm());
  bool end_is_natural = utils::isZero(desired_end_vel.norm());

  if (start_is_natural) {
    val_c_[0] /= val_b_[0];
    val_d_[0] /= val_b_[0];
  } else {
    val_c_[0] = 0;
    val_d_[0] = desired_start_vel.normalized();
  }
  coeff_k_.clear();
  coeff_k_.emplace_back(NJointWaypointDataType::Zero());
  for (size_t i = 1; i < n_coeff_ - 1; ++i) {
    double denominator = val_b_[i] - val_a_[i] * val_c_[i - 1];
    val_c_[i] /= denominator;
    val_d_[i] = (val_d_[i] - val_a_[i] * val_d_[i - 1]) / denominator;
    coeff_k_.emplace_back(NJointWaypointDataType::Zero());
  }

  if (end_is_natural) {
    size_t end_idx = n_coeff_ - 1;
    double denominator =
        val_b_[end_idx] - val_a_[end_idx] * val_c_[end_idx - 1];
    val_c_[end_idx] /= denominator;

    coeff_k_.emplace_back(
        (val_d_[end_idx] - val_a_[end_idx] * val_d_[end_idx - 1]) /
        (val_b_[end_idx] - val_a_[end_idx] * val_c_[end_idx - 1]));
    // coeff_k_.back().normalize();
  } else {
    coeff_k_.emplace_back(desired_end_vel.normalized());
  }

  // Do not use size_t or unsigned int in the following!!!
  for (int i = n_coeff_ - 2; i >= 0; --i) {
    coeff_k_[i] = val_d_[i] - val_c_[i] * coeff_k_[i + 1];
  }
  coeff_k_[0] = start_is_natural ? coeff_k_[0] : desired_start_vel.normalized();
}

template <size_t N_JOINTS, size_t N_MAX_WAYPOINTS, size_t N_MAX_STEPS>
template <typename PathContainer>
void Preprocessing<N_JOINTS, N_MAX_WAYPOINTS, N_MAX_STEPS>::
    computeCoefficientsIterarive(const Waypoints<N_JOINTS>& wps,
                                 PathContainer& s_wps) {
  int n_coeff = n_coeff_;

  NJointWaypointDataType desired_start_vel = wps[0].joints.velocity;
  NJointWaypointDataType desired_end_vel = wps[n_coeff - 1].joints.velocity;
  bool start_is_natural = utils::isZero(desired_start_vel.norm());
  bool end_is_natural = utils::isZero(desired_end_vel.norm());

  for (int i_wp = n_coeff - 2; i_wp >= 0; --i_wp) {
    prepareCoeffMatrix(wps, s_wps);

    if (start_is_natural) {
      val_c_[0] /= val_b_[0];
      val_d_[0] /= val_b_[0];
    } else {
      val_c_[0] = 0;
      val_d_[0] = desired_start_vel.normalized();
    }

    coeff_k_.clear();
    coeff_k_.emplace_back(NJointWaypointDataType::Zero());
    for (size_t i = 1; i < n_coeff_ - 1; ++i) {
      double denominator = val_b_[i] - val_a_[i] * val_c_[i - 1];
      val_c_[i] /= denominator;
      val_d_[i] = (val_d_[i] - val_a_[i] * val_d_[i - 1]) / denominator;
      coeff_k_.emplace_back(NJointWaypointDataType::Zero());
    }

    if (end_is_natural) {
      size_t end_idx = n_coeff - 1;
      double denominator =
          val_b_[end_idx] - val_a_[end_idx] * val_c_[end_idx - 1];
      val_c_[end_idx] /= denominator;

      coeff_k_.emplace_back(
          (val_d_[end_idx] - val_a_[end_idx] * val_d_[end_idx - 1]) /
          (val_b_[end_idx] - val_a_[end_idx] * val_c_[end_idx - 1]));
      // coeff_k_.back().normalize();
    } else {
      coeff_k_.emplace_back(desired_end_vel.normalized());
    }

    // Do not use size_t or unsigned int in the following!!!
    for (int i = n_coeff - 2; i >= 0; --i) {
      coeff_k_[i] = val_d_[i] - val_c_[i] * coeff_k_[i + 1];
    }

    double path_correction =
        (coeff_k_[i_wp].norm() - 1) * (s_wps[i_wp + 1] - s_wps[i_wp]);
    for (int i = i_wp + 1; i < n_coeff; ++i) {
      s_wps[i] += path_correction * 0.5;
    }
  }

  computeCoefficients(wps, s_wps);
}

template <size_t N_JOINTS, size_t N_MAX_WAYPOINTS, size_t N_MAX_STEPS>
template <typename PathContainer>
void Preprocessing<N_JOINTS, N_MAX_WAYPOINTS, N_MAX_STEPS>::prepareCoeffMatrix(
    const Waypoints<N_JOINTS>& wps, PathContainer& s_wps) {
  double one_third = 1.0 / 3.0;
  double one_sixth = 1.0 / 6.0;
  double diff_s = s_wps[1] - s_wps[0];
  double diff_s_2 = diff_s * diff_s;
  NJointWaypointDataType diff_wp =
      wps[1].joints.position - wps[0].joints.position;

  val_a_.clear();
  val_b_.clear();
  val_c_.clear();
  val_d_.clear();

  val_a_.emplace_back(.0);
  val_b_.emplace_back(one_third);
  val_c_.emplace_back(one_sixth);
  val_d_.emplace_back(diff_wp / diff_s * 0.5);

  // Fill coefficients with the right value
  for (size_t i = 1; i < n_coeff_ - 1; ++i) {
    auto diff_wp_next = wps[i + 1].joints.position - wps[i].joints.position;
    double diff_s_next = s_wps[i + 1] - s_wps[i];
    double diff_s_next_2 = diff_s_next * diff_s_next;

    val_a_.emplace_back(one_sixth / diff_s);
    val_b_.emplace_back(one_third / diff_s + one_third / diff_s_next);
    val_c_.emplace_back(one_sixth / (diff_s_next));
    val_d_.emplace_back(0.5 *
                        (diff_wp / diff_s_2 + diff_wp_next / diff_s_next_2));

    diff_s = diff_s_next;
    diff_s_2 = diff_s_next_2;
    diff_wp = diff_wp_next;
  }
  val_a_.emplace_back(one_sixth);
  val_b_.emplace_back(one_third);
  val_c_.emplace_back(.0);
  val_d_.emplace_back(diff_wp / diff_s * 0.5);
}

template <size_t N_JOINTS, size_t N_MAX_WAYPOINTS, size_t N_MAX_STEPS>
template <typename PathContainer, typename PathContainer2>
Waypoints<N_JOINTS>
Preprocessing<N_JOINTS, N_MAX_WAYPOINTS, N_MAX_STEPS>::interpolateBasic(
    const Waypoints<N_JOINTS>& wps, PathContainer& s_wps,
    PathContainer2& s_adjusted_wps, double res) {
  // Prepare the container
  const size_t n_wps = wps.size();
  size_t active_seg = 0;
  double curr_s = .0;
  double adjusted_s = .0;
  double tau = .0;
  double curr_segment = s_wps[active_seg];
  double next_segment = s_wps[active_seg + 1];
  double d_s = next_segment - curr_segment;
  Waypoints<N_JOINTS> interpolated_wps = Waypoints<N_JOINTS>();
  interpolated_wps.reserve(2 * n_wps);

  NJointWaypointDataType curr_pos = wps[active_seg].joints.position;
  NJointWaypointDataType next_pos = wps[active_seg + 1].joints.position;
  NJointWaypointDataType d_pos = next_pos - curr_pos;
  NJointWaypointDataType interpolant = curr_pos;

  NJointWaypointDataType a_i = coeff_k_[active_seg] * d_s - d_pos;
  NJointWaypointDataType b_i = -coeff_k_[active_seg + 1] * d_s + d_pos;

  s_adjusted_wps.clear();
  s_adjusted_wps.emplace_back(.0);
  interpolated_wps.emplace_back(wps[active_seg]);

  // double last_wp_s = 0;
  // bool after_wp_not_reached = true;
  // bool before_wp_not_reached = false;

  while (true) {
    // compute the coefficients
    curr_s += res;
    tau = (curr_s - curr_segment) / d_s;
    while (tau > 1) {
      ++active_seg;
      NJointWaypointDataType diff =
          wps[active_seg].joints.position - interpolant;
      interpolant = wps[active_seg].joints.position;
      adjusted_s += diff.norm();
      s_adjusted_wps.emplace_back(adjusted_s);
      interpolated_wps.emplace_back(wps[active_seg]);
      if (active_seg >= n_wps - 1) {
        break;
      }

      curr_segment = s_wps[active_seg];
      next_segment = s_wps[active_seg + 1];
      d_s = next_segment - curr_segment;

      curr_pos = wps[active_seg].joints.position;
      next_pos = wps[active_seg + 1].joints.position;

      d_pos = next_pos - curr_pos;

      a_i = coeff_k_[active_seg] * d_s - d_pos;
      b_i = -coeff_k_[active_seg + 1] * d_s + d_pos;
      tau = (curr_s - curr_segment) / d_s;
      // after_wp_not_reached = true;
      // last_wp_s = curr_s;
    }

    if (active_seg >= n_wps - 1) {
      break;
    }

    // Compute the interpolated waypoints and update the adjusted s
    // ToDo Xi: Check if these eqations produce the same result as the ones in
    // matlab
    double tau2 = tau * tau;
    double tau3 = tau2 * tau;
    NJointWaypointDataType interpolant_old = interpolant;
    interpolant = curr_pos + tau * (a_i + d_pos) + tau2 * (b_i - 2 * a_i) +
                  (a_i - b_i) * tau3;
    adjusted_s += (interpolant - interpolant_old).norm();

    // // Extend waypoints if we are halfway between waypoints
    // if (after_wp_not_reached &&
    //     curr_s - last_wp_s > d_s * 0.25   ) {
    //   after_wp_not_reached = false;
    //   Waypoint<N_JOINTS> interpl_wp = Waypoint<N_JOINTS>();
    //   interpl_wp.joints.position = interpolant;
    //   interpolated_wps.emplace_back(interpl_wp);
    //   s_adjusted_wps.emplace_back(adjusted_s);
    //   before_wp_not_reached = true;
    // }

    // if (before_wp_not_reached &&
    //     curr_s - last_wp_s > d_s * 0.75  ) {
    //   before_wp_not_reached = false;
    //   Waypoint<N_JOINTS> interpl_wp = Waypoint<N_JOINTS>();
    //   interpl_wp.joints.position = interpolant;
    //   interpolated_wps.emplace_back(interpl_wp);
    //   s_adjusted_wps.emplace_back(adjusted_s);
    // }
  }  // End while
  n_coeff_ = std::min(interpolated_wps.size(), N_MAX_WAYPOINTS);
  return interpolated_wps;
}

template <size_t N_JOINTS, size_t N_MAX_WAYPOINTS, size_t N_MAX_STEPS>
template <typename PathContainer, typename PathContainer2>
void Preprocessing<N_JOINTS, N_MAX_WAYPOINTS, N_MAX_STEPS>::interpolateFinal(
    const Waypoints<N_JOINTS>& wps, PathContainer& s_wps,
    PathContainer2& s_adjusted_final, double res, bool keep_wps) {
  // Prepare the container
  size_t n_wps = wps.size();
  size_t active_seg = 0;
  double curr_s = .0;
  double adjusted_s = .0;
  double tau = .0;
  double curr_segment = s_wps[active_seg];
  double next_segment = s_wps[active_seg + 1];
  double half_res = res * 0.5;
  double d_s = next_segment - curr_segment;
  double d_s2 = d_s * d_s;
  double d_s3 = d_s * d_s2;
  double s_diff = .0;
  bool reducing_res = false;
  // keep_wps = false;

  NJointWaypointDataType curr_pos = wps[active_seg].joints.position;
  NJointWaypointDataType next_pos = wps[active_seg + 1].joints.position;
  NJointWaypointDataType d_pos = next_pos - curr_pos;
  NJointWaypointDataType interpolant = curr_pos;

  NJointWaypointDataType a_i = coeff_k_[active_seg] * d_s - d_pos;
  NJointWaypointDataType b_i = -coeff_k_[active_seg + 1] * d_s + d_pos;

  // TODO(Xi) : Should we clear it here or in the function in the high level?
  s_adjusted_final.clear();
  interpl_.clear();
  derivatives_.clear();
  s_adjusted_final.emplace_back(.0);
  tau_.emplace_back(.0);
  interpl_.emplace_back(wps[active_seg]);
  wp_s_idx_.emplace_back(0);

  DerivativesDataType der_start;
  der_start.first = (a_i + d_pos) / d_s;
  der_start.second = 2 * (b_i - 2 * a_i) / d_s2;
  der_start.third = 6 * (a_i - b_i) / d_s3;
  derivatives_.emplace_back(der_start);

  while (true) {
    curr_s += half_res;
    tau = (curr_s - curr_segment) / d_s;

    while (tau > 1) {
      ++active_seg;
      if (keep_wps) {
        // ToDo: We need to check the distance to the last interpolant
        double dist_to_wp =
            (wps[active_seg].joints.position - interpolant).norm();

        if (dist_to_wp < 0.1 * res) {
          // The last interpolant is very close to waypoint ->
          // The last tau is very close to 1.0
          // Re-arrange the interpolations around waypoint
          size_t wps_idx = interpl_.size() - 1;
          double updated_previous_tau =
              0.5 * (tau_[wps_idx] + tau_[wps_idx - 1]);
          tau_[wps_idx] = updated_previous_tau;

          double updated_previous_tau2 =
              updated_previous_tau * updated_previous_tau;
          double updated_previous_tau3 =
              updated_previous_tau2 * updated_previous_tau;

          // Compute the new interpolated and derivatives with updated tau
          // New interpolant should be shifted away from the waypoint -> higher
          // resolution
          NJointWaypointDataType updated_previous_interpolant =
              curr_pos + updated_previous_tau * (a_i + d_pos) +
              updated_previous_tau2 * (b_i - 2 * a_i) +
              (a_i - b_i) * updated_previous_tau3;
          // Update the data container
          Waypoint<N_JOINTS> previous_interpl_wp = Waypoint<N_JOINTS>();
          previous_interpl_wp.joints.position = updated_previous_interpolant;
          interpl_[wps_idx] = previous_interpl_wp;

          // Compute the derivatives of the updated interpolant
          DerivativesDataType previous_der;
          previous_der.first =
              (a_i + d_pos + 2 * (b_i - 2 * a_i) * updated_previous_tau +
               3 * (a_i - b_i) * updated_previous_tau2) /
              d_s;
          previous_der.second =
              (2 * (b_i - 2 * a_i) + 6 * (a_i - b_i) * updated_previous_tau) /
              d_s2;
          previous_der.third = 6 * (a_i - b_i) / d_s3;

          // Update the data container
          derivatives_[wps_idx] = previous_der;

          adjusted_s = s_adjusted_final[wps_idx - 1] +
                       (interpl_[wps_idx - 1].joints.position -
                        updated_previous_interpolant)
                           .norm();
          s_adjusted_final[wps_idx] = adjusted_s;
          dist_to_wp =
              (wps[active_seg].joints.position - updated_previous_interpolant)
                  .norm();
          reducing_res = true;
        }

        // Add waypoint into the data containers.
        adjusted_s += dist_to_wp;
        interpolant = wps[active_seg].joints.position;
        s_adjusted_final.emplace_back(adjusted_s);
        interpl_.emplace_back(wps[active_seg]);
        tau_.emplace_back(.0);
        size_t wps_idx = interpl_.size() - 1;
        wp_s_idx_.emplace_back(wps_idx);
        // Compute the derivatives
        DerivativesDataType der;
        der.first = (a_i + d_pos + 2 * (b_i - 2 * a_i) + 3 * (a_i - b_i)) / d_s;
        der.second = (2 * (b_i - 2 * a_i) + 6 * (a_i - b_i)) / d_s2;
        der.third = 6 * (a_i - b_i) / d_s3;

        derivatives_.emplace_back(der);
      }
      if (active_seg >= n_wps - 1) {
        break;
      }

      // Update delta s and delta position
      curr_segment = s_wps[active_seg];
      next_segment = s_wps[active_seg + 1];
      d_s = next_segment - curr_segment;
      d_s2 = d_s * d_s;
      d_s3 = d_s * d_s2;

      curr_pos = wps[active_seg].joints.position;
      next_pos = wps[active_seg + 1].joints.position;
      d_pos = next_pos - curr_pos;

      // Coefficients for compuatation of derivaties for this segment
      a_i = coeff_k_[active_seg] * d_s - d_pos;
      b_i = -coeff_k_[active_seg + 1] * d_s + d_pos;
      tau = (curr_s - curr_segment) / d_s;

      s_diff = .0;
    }

    if (active_seg >= n_wps - 1) {
      break;
    }

    // Compute teh waypoints and the derivatives
    // ToDo Xi: Check if these eqations produce the same result as the ones in
    // matlab
    double tau2 = tau * tau;
    double tau3 = tau2 * tau;
    NJointWaypointDataType interpolant_old = interpolant;
    interpolant = curr_pos + tau * (a_i + d_pos) + tau2 * (b_i - 2 * a_i) +
                  (a_i - b_i) * tau3;

    double iterp_diff = (interpolant - interpolant_old).norm();
    s_diff += iterp_diff;

    // Correct the ratio tau to make the segment close to the resolution
    if (s_diff > res || (reducing_res && s_diff > half_res)) {
      // TODO(Xi): Do we have to correct it if the resolution is reduced around
      // the waypoint?
      double correction =
          reducing_res ? .0 : (s_diff - res) / d_s / (iterp_diff / half_res);
      reducing_res = false;
      tau = correction < tau ? tau - correction : tau;

      curr_s = tau * d_s + curr_segment;

      tau2 = tau * tau;
      tau3 = tau2 * tau;
      interpolant = curr_pos + tau * (a_i + d_pos) + tau2 * (b_i - 2 * a_i) +
                    (a_i - b_i) * tau3;

      Waypoint<N_JOINTS> interpl_wp = Waypoint<N_JOINTS>();
      interpl_wp.joints.position = interpolant;

      s_diff -= iterp_diff;
      iterp_diff = (interpolant - interpolant_old).norm();
      s_diff += iterp_diff;
      adjusted_s += s_diff;

      // Compute the derivatives
      DerivativesDataType der;
      der.first =
          (a_i + d_pos + 2 * (b_i - 2 * a_i) * tau + 3 * (a_i - b_i) * tau2) /
          d_s;
      der.second = (2 * (b_i - 2 * a_i) + 6 * (a_i - b_i) * tau) / d_s2;
      der.third = 6 * (a_i - b_i) / d_s3;

      // TODO(Xi): Can we find a more meaningful constant to avoid zero
      // derivatives?
      if (der.first.norm() > 2 * res) {
        interpl_.emplace_back(interpl_wp);
        s_adjusted_final.emplace_back(adjusted_s);
        derivatives_.emplace_back(der);
        tau_.emplace_back(tau);
      }
      s_diff = .0;
    }
  }  // End While

  n_seg_ = interpl_.size();
}
}  // namespace rttopp
