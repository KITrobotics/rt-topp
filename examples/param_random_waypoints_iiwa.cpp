#include <chrono>
#include <fstream>
#include <iomanip>

#include <rttopp/demo_trajectories.h>
#include <rttopp/rttopp2.h>

constexpr auto N_WAYPOINTS = 30;

int main(int argc, char** argv) {
  const size_t n_joints = rttopp::demo_trajectories::NUM_IIWA_JOINTS;
  rttopp::RTTOPP2<n_joints> topp;

  rttopp::Constraints<n_joints> constraints;

  constraints.joints =
      rttopp::demo_trajectories::generateIIWAJointConstraints();
  // asymmetric also works
  // rttopp::demo_trajectories::generateAsymmetricJointConstraints();

  auto waypoints =
      rttopp::demo_trajectories::generateRandomJointWaypoints<n_joints>(
          N_WAYPOINTS, rttopp::demo_trajectories::iiwaJointPositionLimits());

  // set a small velocity at start and end
  waypoints.front().joints.velocity =
      0.3 * rttopp::WaypointJointDataType<n_joints>::Ones();
  waypoints.back().joints.velocity =
      0.3 * rttopp::WaypointJointDataType<n_joints>::Ones();

  rttopp::demo_trajectories::initPerf();

  const auto t1 = std::chrono::high_resolution_clock::now();
  rttopp::Result result = topp.parameterizeFull(constraints, waypoints);
  const auto t2 = std::chrono::high_resolution_clock::now();
  if (result.error()) {
    std::cout << "error in topp run: " << result.message() << std::endl;
    return EXIT_FAILURE;
  }

  result = topp.verifyTrajectory(true);
  if (result.error()) {
    return EXIT_FAILURE;
  }

  std::cout << "topp run successfull" << std::endl;
  std::cout
      << "computation time "
      << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
      << " microseconds" << std::endl;

  if (argc > 1 && std::string(argv[1]) == "json") {
    nlohmann::json j = topp.toJson(waypoints);

    const std::string dir_base = "./../data/";
    const std::string dir = dir_base + "random_waypoints_iiwa";
    const auto output_path = dir + "/param_random_waypoints_" +
                             std::to_string(N_WAYPOINTS) + ".json";

    std::ofstream of(output_path);

    if (!of.is_open()) {
      std::cout << "Could not write to file: " << output_path << std::endl;
      return EXIT_FAILURE;
    }

    of << std::setw(4) << j << std::endl;
  }

  return EXIT_SUCCESS;
}
