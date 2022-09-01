#include <iomanip>

#include <rttopp/demo_trajectories.h>
#include <rttopp/rttopp2.h>

constexpr auto N_WAYPOINTS = 5;

// TODO(wolfgang): increase test cases, add benchmarking

int main() {
  const size_t n_joints = 6;
  rttopp::RTTOPP2<n_joints> topp;
  rttopp::Trajectory<n_joints> trajectory;

  rttopp::Constraints<n_joints> constraints;
  constraints.joints =
      rttopp::demo_trajectories::generateGenericJointConstraints<n_joints>();

  auto waypoints =
      rttopp::demo_trajectories::generateRandomJointWaypoints<n_joints>(
          N_WAYPOINTS,
          rttopp::demo_trajectories::genericJointPositionLimits<n_joints>());

  rttopp::Result result = topp.parameterizeFull(constraints, waypoints);
  if (result.error()) {
    std::cout << "error in full param: " << result.message() << std::endl;
    return EXIT_FAILURE;
  }

  nlohmann::json j = topp.toJson(waypoints);
  const std::string dir_base = "./../data/";
  const std::string dir = dir_base + "random_waypoints_generic_sampling";
  const auto output_path = dir + "/param_random_waypoints_full_" +
                           std::to_string(N_WAYPOINTS) + "_" +
                           std::to_string(0) + ".json";
  std::ofstream of(output_path);
  if (!of.is_open()) {
    std::cout << "Could not write to file: " << output_path << std::endl;
    return EXIT_FAILURE;
  }
  of << std::setw(4) << j << std::endl;

  result = topp.initParameterization(constraints, waypoints);
  if (result.error()) {
    std::cout << "error in topp initParam: " << result.message() << std::endl;
    return EXIT_FAILURE;
  }

  result = topp.sampleFull(&trajectory);
  if (result.error()) {
    std::cout << "error in topp sampleFull: " << result.message() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "sampling successful, trajectory length "
            << trajectory.back().time << "s" << std::endl;

  result = topp.verifyTrajectory(trajectory, true);
  if (result.error()) {
    std::cout << "error in trajectory verify" << std::endl;
  }

  j = topp.toJson(waypoints, trajectory);

  const auto output_path_2 = dir + "/param_random_waypoints_" +
                             std::to_string(N_WAYPOINTS) + "_" +
                             std::to_string(0) + ".json";

  std::ofstream of2(output_path_2);

  if (!of2.is_open()) {
    std::cout << "Could not write to file: " << output_path_2 << std::endl;
    return EXIT_FAILURE;
  }

  of2 << std::setw(4) << j << std::endl;

  if (result.error()) {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
