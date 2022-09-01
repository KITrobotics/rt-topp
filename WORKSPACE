workspace(name = "rttopp")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")


# Google test
# Get it from https://github.com/google/benchmark/blob/main/WORKSPACE#L13
git_repository(
    name = "com_google_googletest",
    remote = "https://github.com/google/googletest.git",
    tag = "release-1.11.0",
)

git_repository(
    name = "com_google_benchmark",
    remote = "https://github.com/google/benchmark.git",
    tag = "v1.6.1",
)

http_archive(
    name = "com_gitlab_libeigen_eigen",
    sha256 = "0215c6593c4ee9f1f7f28238c4e8995584ebf3b556e9dbf933d84feb98d5b9ef",
    strip_prefix = "eigen-3.3.8",
    urls = [
        "https://gitlab.com/libeigen/eigen/-/archive/3.3.8/eigen-3.3.8.tar.bz2",
    ],
    build_file_content =
"""
# See https://github.com/ceres-solver/ceres-solver/issues/337.
cc_library(
    name = 'eigen',
    srcs = [],
    includes = ['.'],
    hdrs = glob(['Eigen/**',
                'unsupported/Eigen/**']),
    visibility = ['//visibility:public'],
)
"""
)

# Boost
new_local_repository(
    name = "boost",
    path = "/usr/include/boost",
    build_file = "external/boost.BUILD",
)

# json
new_local_repository(
    name = "json",
    path = "external/json",
    build_file = "external/json.BUILD",
)
