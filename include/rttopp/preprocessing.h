#pragma once

#include <boost/container/static_vector.hpp>
#include <fstream>
#include <rttopp/types_utils.h>

namespace rttopp {

template <size_t N_JOINTS, size_t N_MAX_WAYPOINTS, size_t N_MAX_STEPS>
class Preprocessing {
 public:
  explicit Preprocessing(const double path_resolution = 0.05,
                         const size_t steps_per_cycle = 0)
      : path_resolution_(path_resolution), steps_per_cycle_(steps_per_cycle) {}

  ~Preprocessing() = default;

  using NJonitWaypointDataType = WaypointJointDataType<N_JOINTS>;
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
   * @brief Compute the partial derivatives given a position along the path `s`.
   *
   * @param s_pos The position of path
   * @param path_derivatives Returned partial derivatives of this position
   */
  JointPathDerivatives<N_JOINTS> computeDerivatives(double s_pos);

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
   * @param s The position of on the one-dimensional path
   * @param jnt_pos The encoded joint position at this s
   */
  void getJointPositionFromPath(double s, NJonitWaypointDataType& jnt_pos);

  [[nodiscard]] WaypointJointDataType<N_JOINTS> getJointPositionFromPath(
      const size_t idx) const {
    assert(idx < n_seg_);
    return interpl_[idx].joints.position;
  };

  [[nodiscard]] size_t getSegmentSize() const { return interpl_.size(); }

  [[nodiscard]] size_t getDerivativesSize() const {
    return derivatives_.size();
  }

  void outputDataAsCSV(const std::string& file_path = "/tmp/spline.csv");

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
  void interpolateBasic(const Waypoints<N_JOINTS>& wps, PathContainer& s_wps,
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

  using DoubleCoeffContainer =
      boost::container::static_vector<double, N_MAX_WAYPOINTS>;
  using MatrixCoeffContainer =
      boost::container::static_vector<NJonitWaypointDataType, N_MAX_WAYPOINTS>;
  using DerivativesDataType = JointPathDerivatives<N_JOINTS>;

  boost::container::static_vector<Waypoint<N_JOINTS>, N_MAX_WAYPOINTS> wps_;
  boost::container::static_vector<Waypoint<N_JOINTS>, N_MAX_STEPS> interpl_;

  MatrixCoeffContainer val_d_;
  MatrixCoeffContainer coeff_k_;
  boost::container::static_vector<DerivativesDataType, N_MAX_STEPS>
      derivatives_;

  size_t n_coeff_;
  size_t n_seg_;

  const double path_resolution_;
  const double steps_per_cycle_;

  DoubleCoeffContainer val_a_;
  DoubleCoeffContainer val_b_;
  DoubleCoeffContainer val_c_;

  boost::container::static_vector<double, N_MAX_WAYPOINTS> s_wps_;
  boost::container::static_vector<double, N_MAX_STEPS> s_;
  boost::container::static_vector<double, N_MAX_STEPS> s_intermediate_;

};  // End of the class declaration.

template <size_t N_JOINTS, size_t N_MAX_WAYPOINTS, size_t N_MAX_STEPS>
size_t Preprocessing<N_JOINTS, N_MAX_WAYPOINTS, N_MAX_STEPS>::processWaypoints(
    const Waypoints<N_JOINTS>& wps) {
  auto n_input_wps = wps.size();
  n_coeff_ = n_input_wps > N_MAX_WAYPOINTS ? N_MAX_WAYPOINTS : n_input_wps;

  // TODO(wolfgang): shouldn't waypoints be always cleared here at the
  // beginning?
  wps_.clear();

  // Save the original waypoint
  for (size_t i = 0; i < n_coeff_; ++i) {
    wps_.emplace_back(wps[i]);
  }

  // Compute the coefficients
  computePathProjectionBasic(wps, s_wps_);

  // TODO(wolfgang): resize is O(N), can't this have this size directly in the
  // declaration or use an array?
  // coeff_k_.resize(N_MAX_WAYPOINTS);
  computeCoefficients(wps, s_wps_);

  // Interpolate for the first time
  interpolateBasic(wps, s_wps_, s_intermediate_, path_resolution_ * 2.0);

  // Compute the coefficent with adjusted path vairable s
  computeCoefficients(wps, s_intermediate_);

  // Interpolate for the second time with
  interpolateFinal(wps, s_intermediate_, s_, path_resolution_);
  return n_seg_;
}

template <size_t N_JOINTS, size_t N_MAX_WAYPOINTS, size_t N_MAX_STEPS>
JointPathDerivatives<N_JOINTS>
Preprocessing<N_JOINTS, N_MAX_WAYPOINTS, N_MAX_STEPS>::computeDerivatives(
    double /*s_pos*/) {
  ;
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
template <typename PathContainer>
void Preprocessing<N_JOINTS, N_MAX_WAYPOINTS, N_MAX_STEPS>::computeCoefficients(
    const Waypoints<N_JOINTS>& wps, PathContainer& s_wps) {
  prepareCoeffMatrix(wps, s_wps);
  size_t n_wps = wps.size();

  // ToDo Xi: USE THE INPUT VELOCITY
  NJonitWaypointDataType desired_start_vel = wps[0].joints.velocity;
  NJonitWaypointDataType desired_end_vel = wps[n_wps - 1].joints.velocity;

  bool start_is_natural = utils::isZero(desired_start_vel.norm());
  bool end_is_natural = utils::isZero(desired_end_vel.norm());

  if (start_is_natural) {
    val_c_[0] /= val_b_[0];
    val_d_[0] /= val_b_[0];
  } else {
    val_c_[0] = 0;
    val_d_[0] = desired_start_vel;
  }
  coeff_k_.clear();
  coeff_k_.emplace_back(NJonitWaypointDataType::Zero());
  for (size_t i = 1; i < n_coeff_ - 1; ++i) {
    double numerator = val_b_[i] - val_a_[i] * val_c_[i - 1];
    val_c_[i] /= numerator;
    val_d_[i] = (val_d_[i] - val_a_[i] * val_d_[i - 1]) / numerator;
    coeff_k_.emplace_back(NJonitWaypointDataType::Zero());
  }

  if (end_is_natural) {
    size_t end_idx = n_coeff_ - 1;
    double numerator = val_b_[end_idx] - val_a_[end_idx] * val_c_[end_idx - 1];
    val_c_[end_idx] /= numerator;
    val_d_[end_idx] =
        (val_d_[end_idx] - val_a_[end_idx] * val_d_[end_idx - 1]) / numerator;

    coeff_k_.emplace_back(
        (val_d_[end_idx] - val_a_[end_idx] * val_d_[end_idx - 1]) /
        (val_b_[end_idx] - val_a_[end_idx] * val_c_[end_idx - 1]));
  } else {
    coeff_k_.emplace_back(desired_end_vel);
  }

  // Do not use size_t or unsigned int in the following!!!
  for (int i = n_coeff_ - 2; i >= 0; --i) {
    coeff_k_[i] = val_d_[i] - val_c_[i] * coeff_k_[i + 1];
  }
  coeff_k_[0] = start_is_natural ? coeff_k_[0] : desired_start_vel;
}

template <size_t N_JOINTS, size_t N_MAX_WAYPOINTS, size_t N_MAX_STEPS>
template <typename PathContainer>
void Preprocessing<N_JOINTS, N_MAX_WAYPOINTS, N_MAX_STEPS>::prepareCoeffMatrix(
    const Waypoints<N_JOINTS>& wps, PathContainer& s_wps) {
  double one_third = 1.0 / 3.0;
  double one_sixth = 1.0 / 6.0;
  double diff_s = s_wps[1] - s_wps[0];
  double diff_s_2 = diff_s * diff_s;
  NJonitWaypointDataType diff_wp =
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
void Preprocessing<N_JOINTS, N_MAX_WAYPOINTS, N_MAX_STEPS>::interpolateBasic(
    const Waypoints<N_JOINTS>& wps, PathContainer& s_wps,
    PathContainer2& s_adjusted, double res) {
  // Prepare the container
  const size_t n_wps = wps.size();
  size_t active_seg = 0;
  double curr_s = .0;
  double adjusted_s = .0;
  double tau = .0;
  double curr_segment = s_wps[active_seg];
  double next_segment = s_wps[active_seg + 1];
  double d_s = next_segment - curr_segment;

  NJonitWaypointDataType curr_pos = wps[active_seg].joints.position;
  NJonitWaypointDataType next_pos = wps[active_seg + 1].joints.position;
  NJonitWaypointDataType d_pos = next_pos - curr_pos;
  NJonitWaypointDataType interpolant = curr_pos;

  NJonitWaypointDataType a_i = coeff_k_[active_seg] * d_s - d_pos;
  NJonitWaypointDataType b_i = -coeff_k_[active_seg + 1] * d_s + d_pos;

  s_adjusted.clear();
  s_adjusted.emplace_back(.0);
  while (true) {
    // compute the coefficients
    curr_s += res;
    tau = (curr_s - curr_segment) / d_s;
    while (tau > 1) {
      ++active_seg;
      NJonitWaypointDataType diff =
          wps[active_seg].joints.position - interpolant;
      interpolant = wps[active_seg].joints.position;
      adjusted_s += diff.norm();
      s_adjusted.emplace_back(adjusted_s);
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
    }

    if (active_seg >= n_wps - 1) {
      break;
    }

    // Compute the interpolated waypoints and update the adjusted s
    // ToDo Xi: Check if these eqations produce the same result as the ones in
    // matlab
    double tau2 = tau * tau;
    double tau3 = tau2 * tau;
    NJonitWaypointDataType interpolant_old = interpolant;
    interpolant = curr_pos + tau * (a_i + d_pos) + tau2 * (b_i - 2 * a_i) +
                  (a_i - b_i) * tau3;
    adjusted_s += (interpolant - interpolant_old).norm();
  }  // End while
}

template <size_t N_JOINTS, size_t N_MAX_WAYPOINTS, size_t N_MAX_STEPS>
template <typename PathContainer, typename PathContainer2>
void Preprocessing<N_JOINTS, N_MAX_WAYPOINTS, N_MAX_STEPS>::interpolateFinal(
    const Waypoints<N_JOINTS>& wps, PathContainer& s_wps,
    PathContainer2& s_adjusted, double res, bool keep_wps) {
  // Prepare the container
  size_t n_wps = wps.size();
  size_t active_seg = 0;
  double curr_s = .0;
  double adjusted_s = .0;
  double tau = .0;
  double curr_segment = s_wps[active_seg];
  double next_segment = s_wps[active_seg + 1];
  double d_s = next_segment - curr_segment;

  NJonitWaypointDataType curr_pos = wps[active_seg].joints.position;
  NJonitWaypointDataType next_pos = wps[active_seg + 1].joints.position;
  NJonitWaypointDataType d_pos = next_pos - curr_pos;
  NJonitWaypointDataType interpolant = curr_pos;

  NJonitWaypointDataType a_i = coeff_k_[active_seg] * d_s - d_pos;
  NJonitWaypointDataType b_i = -coeff_k_[active_seg + 1] * d_s + d_pos;

  s_adjusted.clear();
  interpl_.clear();
  derivatives_.clear();
  s_adjusted.emplace_back(.0);
  interpl_.emplace_back(wps[active_seg]);

  DerivativesDataType der_start;
  der_start.first = (a_i + d_pos) / d_s;
  der_start.second = 2 * (b_i - 2 * a_i) / d_s / d_s;
  der_start.third = 6 * (a_i - b_i) / d_s / d_s / d_s;
  derivatives_.emplace_back(der_start);

  while (true) {
    curr_s += res;
    tau = (curr_s - curr_segment) / d_s;

    while (tau > 1) {
      ++active_seg;
      if (keep_wps) {
        adjusted_s += (wps[active_seg].joints.position - interpolant).norm();
        interpolant = wps[active_seg].joints.position;

        // s_adjusted.emplace_back(adjusted_s);
        s_adjusted.emplace_back(adjusted_s);
        interpl_.emplace_back(wps[active_seg]);
        // Compute the derivatives
        DerivativesDataType der;
        der.first = (a_i + d_pos + 2 * (b_i - 2 * a_i) + 3 * (a_i - b_i)) / d_s;
        der.second = (2 * (b_i - 2 * a_i) + 6 * (a_i - b_i)) / d_s / d_s;
        der.third = 6 * (a_i - b_i) / d_s / d_s / d_s;
        derivatives_.emplace_back(der);
      }
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
    }

    if (active_seg >= n_wps - 1) {
      break;
    }

    // Compute teh waypoints and the derivatives
    // ToDo Xi: Check if these eqations produce the same result as the ones in
    // matlab
    double tau2 = tau * tau;
    double tau3 = tau2 * tau;
    NJonitWaypointDataType interpolant_old = interpolant;
    interpolant = curr_pos + tau * (a_i + d_pos) + tau2 * (b_i - 2 * a_i) +
                  (a_i - b_i) * tau3;
    Waypoint<N_JOINTS> interpl_wp = Waypoint<N_JOINTS>();
    interpl_wp.joints.position = interpolant;
    interpl_.emplace_back(interpl_wp);

    adjusted_s += (interpolant - interpolant_old).norm();
    // s_adjusted.emplace_back(adjusted_s);
    s_adjusted.emplace_back(adjusted_s);

    // Compute the derivatives
    DerivativesDataType der;
    der.first =
        (a_i + d_pos + 2 * (b_i - 2 * a_i) * tau + 3 * (a_i - b_i) * tau2) /
        d_s;
    der.second = (2 * (b_i - 2 * a_i) + 6 * (a_i - b_i) * tau) / d_s / d_s;
    der.third = 6 * (a_i - b_i) / d_s / d_s / d_s;
    derivatives_.emplace_back(der);
  }  // End While

  n_seg_ = interpl_.size();
}
}  // namespace rttopp
