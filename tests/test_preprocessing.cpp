#include <fstream>
#include <iostream>

#include <gtest/gtest.h>
#include <unsupported/Eigen/Splines>

#include <rttopp/demo_trajectories.h>
#include <rttopp/preprocessing.h>

TEST(RandomPreprocessing, randomWpsPreprocessing) {
  const size_t n_joints = rttopp::demo_trajectories::NUM_IIWA_JOINTS;
  std::array<double, n_joints> jnt_limits;
  for (auto& jnt : jnt_limits) {
    jnt = M_PI;
  }

  const size_t n_waypoints = 10;
  const size_t max_waypoint = 50;
  const size_t max_segments = 3000;
  rttopp::Waypoints<n_joints> wps =
      rttopp::demo_trajectories::generateRandomJointWaypoints(n_waypoints,
                                                              jnt_limits);

  // Construct the instance of the preprocessing
  auto preprocessing =
      rttopp::Preprocessing<n_joints, max_waypoint, max_segments>();
  size_t n_seg = preprocessing.processWaypoints(wps);

  EXPECT_GE(n_seg, n_waypoints) << "The number of the segments should be "
                                   "greater than the number of the waypoints";
  EXPECT_GE(max_segments, n_seg) << "The number of the segments is not allowed "
                                    "to be greater than the max value";
}

TEST(FxiedPreprocessing, fixedWpsPreprocessing) {
  const size_t n_joints = 2;
  const size_t n_waypoints = 10;
  const size_t max_waypoint = 50;
  const size_t max_segments = 3000;
  Eigen::Matrix<double, 10, 2> wps_eigen;
  rttopp::Waypoints<n_joints> wps;
  rttopp::Waypoint<n_joints> p0, p1, p2, p3, p4, p5, p6, p7, p8, p9;
  p0.joints.position << 0.846662407210134, 0.134478322308974;
  p1.joints.position << 0.561165540265034, 0.259977350343059;
  p2.joints.position << 0.454875402405753, 0.0476346761692857;
  p3.joints.position << 0.352174913691962, 0.394293304729078;
  p4.joints.position << 0.585851384134837, 0.371616192088212;
  p5.joints.position << 0.535749744273076, 0.556732923285543;
  p6.joints.position << 0.827456276779346, 0.897494554367658;
  p7.joints.position << 0.264542198722034, 0.167345887006384;
  p8.joints.position << 0.478379061338716, 0.407206419829827;
  p9.joints.position << 0.347785022845722, 0.652353338778751;

  wps.push_back(p0);
  wps.push_back(p1);
  wps.push_back(p2);
  wps.push_back(p3);
  wps.push_back(p4);
  wps.push_back(p5);
  wps.push_back(p6);
  wps.push_back(p7);
  wps.push_back(p8);
  wps.push_back(p9);

  wps_eigen << 0.846662407210134, 0.134478322308974, 0.561165540265034,
      0.259977350343059, 0.454875402405753, 0.047634676169285,
      0.352174913691962, 0.394293304729078, 0.585851384134837,
      0.371616192088212, 0.535749744273076, 0.556732923285543,
      0.827456276779346, 0.897494554367658, 0.264542198722034,
      0.167345887006384, 0.478379061338716, 0.407206419829827,
      0.347785022845722, 0.652353338778751;

  std::ofstream myfile;
  myfile.open("/tmp/eigen_our_spline.csv");

  // Construct the instance of the preprocessing
  auto preprocessing =
      rttopp::Preprocessing<n_joints, max_waypoint, max_segments>(0.05);
  size_t n_seg = preprocessing.processWaypoints(wps);

  EXPECT_GE(n_seg, n_waypoints) << "The number of the segments should be "
                                   "greater than the number of the waypoints";
  EXPECT_GE(max_segments, n_seg) << "The number of the segments is not allowed "
                                    "to be greater than the max value";

  auto n_interpl_seg = preprocessing.getSegmentSize();
  auto n_derivatives_seg = preprocessing.getDerivativesSize();
  EXPECT_EQ(n_interpl_seg, n_seg)
      << "The number of the interpolation segment is"
         "not equal to the return value.";
  EXPECT_EQ(n_derivatives_seg, n_seg) << "The number of the derivatives is"
                                         "not equal to the return.";

  // ToDo Xi: Assert the interpolated value to the matlab outputs.

  //
  using SplineType = Eigen::Spline<double, n_joints>;
  SplineType spline =
      Eigen::SplineFitting<SplineType>::Interpolate(wps_eigen.transpose(), 3);

  const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision,
                                         Eigen::DontAlignCols, ", ", "\n");
  const double max_s = preprocessing.getPathPosition(n_seg - 1);
  for (std::size_t idx = 0; idx < n_seg; ++idx) {
    const auto u = preprocessing.getPathPosition(idx) / max_s;
    const Eigen::MatrixXd derivatives = spline.derivatives(u, 3);
    auto eigen_position = derivatives.col(0);
    auto eigen_vel = derivatives.col(1);
    auto eigen_acc = derivatives.col(2);
    auto our_derivatives = preprocessing.getDerivatives(idx);
    auto s_position = preprocessing.getPathPosition(idx);
    rttopp::WaypointJointDataType<n_joints> our_position =
        preprocessing.getJointPositionFromPath(idx);
    myfile << eigen_position.transpose().format(CSVFormat) << ", "
           << eigen_vel.transpose().format(CSVFormat) << ", "
           << eigen_acc.transpose().format(CSVFormat) << ", "
           << our_position.transpose().format(CSVFormat) << ", "
           << our_derivatives.first.transpose().format(CSVFormat) << ","
           << our_derivatives.second.transpose().format(CSVFormat) << ","
           << s_position << std::endl;
  }
  myfile.close();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
