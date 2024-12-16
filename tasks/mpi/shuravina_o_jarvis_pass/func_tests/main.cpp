#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <core/task/include/task.hpp>
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

TEST(shuravina_o_jarvis_pass, Test_10_Points) {
  boost::mpi::communicator world;
  const int count_points = 10;
  std::vector<Point> global_points = getRandomPoints(count_points, 0, 100);
  std::vector<Point> global_hull;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_points.data()));
    taskDataPar->inputs_count.emplace_back(global_points.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_hull.data()));
    taskDataPar->outputs_count.emplace_back(global_hull.size());
  }

  JarvisPassMPI jarvis_mpi(global_points);
  ASSERT_TRUE(jarvis_mpi.validation());
  jarvis_mpi.run();
  global_hull = jarvis_mpi.get_hull();

  if (world.rank() == 0) {
    std::vector<Point> seq_hull = jarvis_march(global_points);

    ASSERT_EQ(global_hull.size(), seq_hull.size());
    for (size_t i = 0; i < global_hull.size(); i++) {
      EXPECT_EQ(global_hull[i], seq_hull[i]);
    }
  }
}

TEST(shuravina_o_jarvis_pass, Test_100_Points) {
  boost::mpi::communicator world;
  const int count_points = 100;
  std::vector<Point> global_points = getRandomPoints(count_points, 0, 100);
  std::vector<Point> global_hull;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_points.data()));
    taskDataPar->inputs_count.emplace_back(global_points.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_hull.data()));
    taskDataPar->outputs_count.emplace_back(global_hull.size());
  }

  JarvisPassMPI jarvis_mpi(global_points);
  ASSERT_TRUE(jarvis_mpi.validation());
  jarvis_mpi.run();
  global_hull = jarvis_mpi.get_hull();

  if (world.rank() == 0) {
    std::vector<Point> seq_hull = jarvis_march(global_points);

    ASSERT_EQ(global_hull.size(), seq_hull.size());
    for (size_t i = 0; i < global_hull.size(); i++) {
      EXPECT_EQ(global_hull[i], seq_hull[i]);
    }
  }
}

TEST(shuravina_o_jarvis_pass, Test_1000_Points) {
  boost::mpi::communicator world;
  const int count_points = 1000;
  std::vector<Point> global_points = getRandomPoints(count_points, 0, 100);
  std::vector<Point> global_hull;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_points.data()));
    taskDataPar->inputs_count.emplace_back(global_points.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_hull.data()));
    taskDataPar->outputs_count.emplace_back(global_hull.size());
  }

  JarvisPassMPI jarvis_mpi(global_points);
  ASSERT_TRUE(jarvis_mpi.validation());
  jarvis_mpi.run();
  global_hull = jarvis_mpi.get_hull();

  if (world.rank() == 0) {
    std::vector<Point> seq_hull = jarvis_march(global_points);

    ASSERT_EQ(global_hull.size(), seq_hull.size());
    for (size_t i = 0; i < global_hull.size(); i++) {
      EXPECT_EQ(global_hull[i], seq_hull[i]);
    }
  }
}

TEST(shuravina_o_jarvis_pass, Test_Random_Points) {
  boost::mpi::communicator world;
  const int count_points = 500;
  std::vector<Point> global_points = getRandomPoints(count_points, 0, 100);
  std::vector<Point> global_hull;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_points.data()));
    taskDataPar->inputs_count.emplace_back(global_points.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_hull.data()));
    taskDataPar->outputs_count.emplace_back(global_hull.size());
  }

  JarvisPassMPI jarvis_mpi(global_points);
  ASSERT_TRUE(jarvis_mpi.validation());
  jarvis_mpi.run();
  global_hull = jarvis_mpi.get_hull();

  if (world.rank() == 0) {
    std::vector<Point> seq_hull = jarvis_march(global_points);

    ASSERT_EQ(global_hull.size(), seq_hull.size());
    for (size_t i = 0; i < global_hull.size(); i++) {
      EXPECT_EQ(global_hull[i], seq_hull[i]);
    }
  }
}