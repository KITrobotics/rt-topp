#include <gtest/gtest.h>

#include <rttopp/demo_trajectories.h>
#include <rttopp/rttopp2.h>

TEST(ParamRTTOPP2, StartStateVelTooHigh) {
  const size_t n_joints = 6;
  rttopp::RTTOPP2<n_joints> topp;
  rttopp::Constraints<n_joints> constraints;
  rttopp::Waypoints<n_joints> waypoints(5);

  constraints.joints =
      rttopp::demo_trajectories::generateGenericJointConstraints<n_joints>();

  // these are the waypoints from the failing path 9616 in the generic example
  waypoints[0].joints.position << 0.7106334292475558, 2.107637453550443,
      2.010407531616945, 0.5423096137321899, 1.7269491750497759,
      -2.0992195101647413;
  waypoints[1].joints.position << 0.18440064890885655, 1.7976667046460513,
      2.0787004249842083, 0.6097202430116404, 1.5137116683346754,
      -2.115691412107438;
  waypoints[2].joints.position << -0.7384912323434127, 2.481684944410026,
      1.6318946497858908, -2.1117846808575225, 2.06777630179952,
      0.7840938464281209;
  waypoints[3].joints.position << 1.22742646932319, 1.7886828787258562,
      0.2461982557712492, -1.2717105410062175, 1.932068208886463,
      -0.28056745678997164;
  waypoints[4].joints.position << 2.253314331834745, -1.570936405721035,
      -1.7757477319814394, -2.353442807598746, 0.8182044684589265,
      0.06441215492262309;

  waypoints.front().joints.velocity =
      0.4 * rttopp::WaypointJointDataType<n_joints>::Ones();
  waypoints.back().joints.velocity =
      0.4 * rttopp::WaypointJointDataType<n_joints>::Ones();

  rttopp::Result result = topp.parameterizeFull(constraints, waypoints);
  // if (result.error()) {
  //   std::cout << "error in topp run: " << result.message()
  //             << std::endl;
  // }
  //       nlohmann::json j = topp.toJson(waypoints);

  //   const std::string dir_base = "./../data/";
  //   const std::string dir = dir_base + "param_tests";
  //   const auto output_path = dir + "/vel_start.json";

  //   std::ofstream of(output_path);

  //   if (!of.is_open()) {
  //     std::cout << "Could not write to file: " << output_path << std::endl;
  //     return;
  //   }

  //   of << std::setw(4) << j << std::endl;

  EXPECT_TRUE(result.error());
  EXPECT_EQ(result.val(), rttopp::Result::START_STATE_VELOCITY_TOO_HIGH);
}

TEST(ParamRTTOPP2, RandomWaypointsZeroStartGoalVel) {
  constexpr auto N_TRAJECTORIES = 1000 * 1000;
  constexpr auto N_WAYPOINTS = 5;
  const size_t n_joints = 6;

  rttopp::RTTOPP2<n_joints> topp;
  rttopp::Constraints<n_joints> constraints;

  constraints.joints =
      rttopp::demo_trajectories::generateGenericJointConstraints<n_joints>();

  int error_counter = 0;
  for (size_t i = 0; i < N_TRAJECTORIES; ++i) {
    auto waypoints =
        rttopp::demo_trajectories::generateRandomJointWaypoints<n_joints>(
            N_WAYPOINTS,
            rttopp::demo_trajectories::genericJointPositionLimits<n_joints>());

    rttopp::Result result = topp.parameterizeFull(constraints, waypoints);
    if (result.error()) {
      std::cout << "error in topp run " << i << ": " << result.message()
                << std::endl;
      error_counter++;
    }
    EXPECT_TRUE(result.success() ||
                result.val() == rttopp::Result::NOT_SOLVABLE_VELOCITY_STALLING);

    if (result.success()) {
      result = topp.verifyTrajectory();
      if (result.error()) {
        std::cout << "error in trajectory verify " << i << std::endl;
      }
      EXPECT_FALSE(result.error());
    }

    if (result.error()) {
      nlohmann::json j = topp.toJson(waypoints);

      const std::string dir_base = "./../data/";
      const std::string dir = dir_base + "random_waypoints_generic_failed";
      const auto output_path = dir + "/param_random_waypoints_" +
                               std::to_string(N_WAYPOINTS) + "_" +
                               std::to_string(i) + ".json";

      std::ofstream of(output_path);

      of << std::setw(4) << j << std::endl;
    }
  }

  // zero cases with stalling velocity
  EXPECT_EQ(error_counter, 0);
}

TEST(ParamRTTOPP2, RandomWaypointsSmallStartGoalVel) {
  // TODO(wolfgang): path 9616 first case where start velocity too high, likely
  // first two waypoints too close to each other and no room to hold such a
  // velocity.
  // path 39868 has velocity stalling
  constexpr auto N_TRAJECTORIES = 43 * 1000;
  constexpr auto N_WAYPOINTS = 5;
  const size_t n_joints = 6;

  rttopp::RTTOPP2<n_joints> topp;
  rttopp::Constraints<n_joints> constraints;

  constraints.joints =
      rttopp::demo_trajectories::generateGenericJointConstraints<n_joints>();

  int error_counter = 0;
  for (size_t i = 0; i < N_TRAJECTORIES; ++i) {
    auto waypoints =
        rttopp::demo_trajectories::generateRandomJointWaypoints<n_joints>(
            N_WAYPOINTS,
            rttopp::demo_trajectories::genericJointPositionLimits<n_joints>());

    // set a small velocity at start and end
    waypoints.front().joints.velocity =
        0.4 * rttopp::WaypointJointDataType<n_joints>::Ones();
    waypoints.back().joints.velocity =
        0.4 * rttopp::WaypointJointDataType<n_joints>::Ones();

    rttopp::Result result = topp.parameterizeFull(constraints, waypoints);
    if (result.error()) {
      std::cout << "error in topp run " << i << ": " << result.message()
                << std::endl;
      error_counter++;
    }
    EXPECT_TRUE(result.success() ||
                result.val() == rttopp::Result::START_STATE_VELOCITY_TOO_HIGH ||
                result.val() == rttopp::Result::GOAL_STATE_VELOCITY_TOO_HIGH);

    if (result.success()) {
      result = topp.verifyTrajectory();
      if (result.error()) {
        std::cout << "error in trajectory verify " << i << std::endl;
      }
      EXPECT_FALSE(result.error());
    }
  }

  // maximum four cases with start or goal state velocity too high
  EXPECT_EQ(error_counter, 4);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
