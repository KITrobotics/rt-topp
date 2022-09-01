#include <chrono>
#include <fstream>
#include <iomanip>

#include <rttopp/demo_trajectories.h>
#include <rttopp/rttopp2.h>

// When using start/goal velocity 1.0 and checking start/goal state for
// validity, already run 67 has invaid start velocity. when testing with 0.4
// start/goal velocity, path 9616 has a too high start velocity.
constexpr auto N_TRAJECTORIES = 9 * 1000;
constexpr auto N_WAYPOINTS = 5;

int main(int argc, char** argv) {
  const size_t n_joints = 6;
  rttopp::RTTOPP2<n_joints> topp;
  Eigen::VectorXd durations(N_TRAJECTORIES), inf_means(N_TRAJECTORIES),
      inf_std_devs(N_TRAJECTORIES), nums_gridpoints(N_TRAJECTORIES),
      max_normalized_velocities(N_TRAJECTORIES),
      max_normalized_accelerations(N_TRAJECTORIES);

  rttopp::Constraints<n_joints> constraints;
  constraints.joints =
      rttopp::demo_trajectories::generateGenericJointConstraints<n_joints>();

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

    const auto t1 = std::chrono::high_resolution_clock::now();
    rttopp::Result result = topp.parameterizeFull(constraints, waypoints);
    const auto t2 = std::chrono::high_resolution_clock::now();
    if (result.error()) {
      std::cout << "error in topp run " << i << ": " << result.message()
                << std::endl;
    }

    durations[i] =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    // if(i % 10 == 0)
    // std::cout << "topp run " << i << " successfull" << std::endl;
    // std::cout << "computation time " << durations[i]
    //           << " microseconds" << std::endl;

    size_t num_gridpoints;
    result = topp.verifyTrajectory(
        false, &num_gridpoints, &inf_means[i], &inf_std_devs[i],
        &max_normalized_velocities[i], &max_normalized_accelerations[i]);
    nums_gridpoints[i] = num_gridpoints;

    if ((argc > 1 && std::string(argv[1]) == "json") || result.error()) {
      nlohmann::json j = topp.toJson(waypoints);

      const std::string dir_base = "./../data/";
      const std::string dir = dir_base + "random_waypoints_generic";
      const auto output_path = dir + "/param_random_waypoints_" +
                               std::to_string(N_WAYPOINTS) + "_" +
                               std::to_string(i) + ".json";

      std::ofstream of(output_path);

      if (!of.is_open()) {
        std::cout << "error in trajectory " << i << std::endl;
        std::cout << "Could not write to file: " << output_path << std::endl;
        return EXIT_FAILURE;
      }

      of << std::setw(4) << j << std::endl;
    }

    if (result.error()) {
      std::cout << "error in trajectory " << i << std::endl;

      return EXIT_FAILURE;
    }
  }

  std::cout << "all tested " << N_TRAJECTORIES
            << " trajectories passed successfully" << std::endl;
  std::cout << "mean, min, max values:" << std::endl;
  std::cout << "gridpoints: " << nums_gridpoints.mean() << ", "
            << nums_gridpoints.minCoeff() << ", " << nums_gridpoints.maxCoeff()
            << std::endl;
  std::cout << "durations (microseconds): " << durations.mean() << ", "
            << durations.minCoeff() << ", " << durations.maxCoeff()
            << std::endl;
  std::cout << "infinity norm mean: " << inf_means.mean() << ", "
            << inf_means.minCoeff() << ", " << inf_means.maxCoeff()
            << std::endl;
  std::cout << "infinity norm standard deviation: " << inf_std_devs.mean()
            << ", " << inf_std_devs.minCoeff() << ", "
            << inf_std_devs.maxCoeff() << std::endl;
  std::cout << "maximum normalized velocities: "
            << max_normalized_velocities.mean() << ", "
            << max_normalized_velocities.minCoeff() << ", "
            << max_normalized_velocities.maxCoeff() << std::endl;
  std::cout << "maximum normalized accelerations: "
            << max_normalized_accelerations.mean() << ", "
            << max_normalized_accelerations.minCoeff() << ", "
            << max_normalized_accelerations.maxCoeff() << std::endl;

  return EXIT_SUCCESS;
}
