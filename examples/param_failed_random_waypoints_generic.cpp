#include <chrono>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>

#include <rttopp/demo_trajectories.h>
#include <rttopp/rttopp2.h>

// This program reads the failed .json file and compute it again for faster
// development
constexpr auto N_JOINTS = 6;
constexpr auto N_TRAJECTORIES = 100;

template <typename EigenVectorType, typename VectorType>
void fromVectroAsEigenVector(VectorType &v, EigenVectorType &mat) {
  mat = Eigen::Map<EigenVectorType, Eigen::Unaligned>(v.data(), v.size());
}

rttopp::Waypoints<N_JOINTS> fromJson(const nlohmann::json &j) {
  size_t n_waypoints = j["waypoints"].size();
  rttopp::Waypoints<N_JOINTS> waypoints(n_waypoints);

  for (size_t idx = 0; idx < n_waypoints; ++idx) {
    auto joint_positions =
        j["waypoints"][idx]["joints"]["position"].get<std::vector<double>>();

    auto &waypoint = waypoints[idx].joints.position;
    fromVectroAsEigenVector(joint_positions, waypoint);
  }

  return waypoints;
}

int main(int argc, char **argv) {  // NOLINT bugprone-exception-escape
  if (argc == 1) {
    std::cout << "Please give the directory to the .json files. Abort.";
    return EXIT_FAILURE;
  }
  // Todo(Xi): what if we want to run a specific file in the directory??
  rttopp::RTTOPP2<N_JOINTS> topp;
  Eigen::VectorXd durations(N_TRAJECTORIES), inf_means(N_TRAJECTORIES),
      inf_std_devs(N_TRAJECTORIES), nums_gridpoints(N_TRAJECTORIES),
      max_normalized_velocities(N_TRAJECTORIES),
      max_normalized_accelerations(N_TRAJECTORIES);

  // Read Files
  size_t i = 0;
  size_t i_failed = 0;
  std::string path = argv[1];
  for (const auto &entry :
       std::experimental::filesystem::directory_iterator(path)) {
    std::ifstream ifs(entry.path());
    nlohmann::json j = nlohmann::json::parse(ifs);

    auto waypoints = fromJson(j);

    // Set constraints
    rttopp::Constraints<N_JOINTS> constraints;
    constraints.joints =
        rttopp::demo_trajectories::generateGenericJointConstraints<N_JOINTS>();

    // set a small velocity at start and end
    waypoints.front().joints.velocity =
        0.0 * rttopp::WaypointJointDataType<N_JOINTS>::Ones();
    waypoints.back().joints.velocity =
        0.0 * rttopp::WaypointJointDataType<N_JOINTS>::Ones();

    rttopp::Result result = topp.parameterizeFull(constraints, waypoints);
    if (result.error()) {
      std::cout << "Error: " << result.message() << std::endl;
    }

    size_t num_gridpoints;
    rttopp::Result verification = topp.verifyTrajectory(
        false, &num_gridpoints, &inf_means[i], &inf_std_devs[i],
        &max_normalized_velocities[i], &max_normalized_accelerations[i]);

    if (verification.error()) {
      std::cout << "error in trajectory " << entry.path() << std::endl;
      std::cout << "-----------------------\n" << std::endl;
    }

    if (result.error() || verification.error()) {
      ++i_failed;
      nlohmann::json j_out = topp.toJson(waypoints);

      // auto new_path = entry.path();
      // auto file_name = new_path.filename();

      // std::string path_prefix_str = "new_";
      // std::experimental::filesystem::path new_file_name(path_prefix_str);
      // new_file_name += file_name;
      // new_path.replace_filename(new_file_name);

      // std::ofstream of(new_path);
      std::ofstream of(entry.path());

      // ToDo(Xi): Should we overwrite it?
      if (!of.is_open()) {
        std::cout << "Could not write to file: " << entry.path() << std::endl;
        return 0;
      }

      of << std::setw(4) << j_out << std::endl;
    }
    ++i;
  }

  std::cout << "Checked " << i << " cases, " << i_failed << " of them failed."
            << std::endl;

  return 0;
}
