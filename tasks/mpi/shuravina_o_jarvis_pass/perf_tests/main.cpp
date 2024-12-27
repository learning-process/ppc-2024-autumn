#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <chrono>
#include <random>
#include <vector>

#include "mpi/shuravina_o_jarvis_pass/include/ops_mpi.hpp"

using namespace shuravina_o_jarvis_pass;

static std::vector<Point> getRandomPoints(int count, int min_coord, int max_coord) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(min_coord, max_coord);
  std::vector<Point> points(count);
  for (int i = 0; i < count; i++) {
    points[i] = Point(dist(gen), dist(gen));
  }
  return points;
}
TEST(shuravina_o_jarvis_pass_mpi_perf, Test_1000_Points) {
  boost::mpi::communicator world;
  const int count_points = 1000;
  std::vector<Point> global_points = getRandomPoints(count_points, 0, 100);

  JarvisPassMPI jarvis_mpi(global_points);
  ASSERT_TRUE(jarvis_mpi.validation());

  auto start = std::chrono::high_resolution_clock::now();
  jarvis_mpi.run();
  auto end = std::chrono::high_resolution_clock::now();

  if (world.rank() == 0) {
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken for 1000 points (MPI): " << elapsed.count() << " seconds" << std::endl;
  }
}

TEST(shuravina_o_jarvis_pass_mpi_perf, Test_10000_Points) {
  boost::mpi::communicator world;
  const int count_points = 10000;
  std::vector<Point> global_points = getRandomPoints(count_points, 0, 100);

  JarvisPassMPI jarvis_mpi(global_points);
  ASSERT_TRUE(jarvis_mpi.validation());

  auto start = std::chrono::high_resolution_clock::now();
  jarvis_mpi.run();
  auto end = std::chrono::high_resolution_clock::now();

  if (world.rank() == 0) {
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken for 10000 points (MPI): " << elapsed.count() << " seconds" << std::endl;
  }
}