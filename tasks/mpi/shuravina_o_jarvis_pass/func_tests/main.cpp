#include <gtest/gtest.h>
#include <mpi.h>

#include <vector>

#include "mpi/shuravina_o_jarvis_pass/include/ops_mpi.hpp"

using namespace shuravina_o_jarvis_pass;

TEST(shuravina_o_jarvis_pass, Test_Fixed_Points) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<Point> global_points = {Point(0, 0), Point(1, 1), Point(2, 2), Point(0, 2), Point(2, 0)};
  std::vector<Point> global_hull;

  JarvisPassMPI jarvis_mpi(global_points);
  ASSERT_TRUE(jarvis_mpi.validation());
  jarvis_mpi.run();
  global_hull = jarvis_mpi.get_hull();

  if (rank == 0) {
    std::vector<Point> seq_hull = jarvis_march(global_points);
    ASSERT_EQ(global_hull.size(), seq_hull.size());
    for (size_t i = 0; i < global_hull.size(); i++) {
      EXPECT_EQ(global_hull[i], seq_hull[i]);
    }
  }
}

TEST(shuravina_o_jarvis_pass, Test_Minimal_Points) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<Point> global_points = {Point(0, 0), Point(1, 1), Point(0, 2)};
  std::vector<Point> global_hull;

  JarvisPassMPI jarvis_mpi(global_points);
  ASSERT_TRUE(jarvis_mpi.validation());
  jarvis_mpi.run();
  global_hull = jarvis_mpi.get_hull();

  if (rank == 0) {
    std::vector<Point> seq_hull = jarvis_march(global_points);
    ASSERT_EQ(global_hull.size(), seq_hull.size());
    for (size_t i = 0; i < global_hull.size(); i++) {
      EXPECT_EQ(global_hull[i], seq_hull[i]);
    }
  }
}

TEST(shuravina_o_jarvis_pass, Test_Collinear_Points) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<Point> global_points = {Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)};
  std::vector<Point> global_hull;

  JarvisPassMPI jarvis_mpi(global_points);
  ASSERT_TRUE(jarvis_mpi.validation());
  jarvis_mpi.run();
  global_hull = jarvis_mpi.get_hull();

  if (rank == 0) {
    std::vector<Point> seq_hull = jarvis_march(global_points);
    ASSERT_EQ(global_hull.size(), seq_hull.size());
    for (size_t i = 0; i < global_hull.size(); i++) {
      EXPECT_EQ(global_hull[i], seq_hull[i]);
    }
  }
}

TEST(shuravina_o_jarvis_pass, Test_All_Points_On_Hull) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<Point> global_points = {Point(0, 0), Point(0, 2), Point(2, 2), Point(2, 0)};
  std::vector<Point> global_hull;

  JarvisPassMPI jarvis_mpi(global_points);
  ASSERT_TRUE(jarvis_mpi.validation());
  jarvis_mpi.run();
  global_hull = jarvis_mpi.get_hull();

  if (rank == 0) {
    std::vector<Point> seq_hull = jarvis_march(global_points);
    ASSERT_EQ(global_hull.size(), seq_hull.size());
    for (size_t i = 0; i < global_hull.size(); i++) {
      EXPECT_EQ(global_hull[i], seq_hull[i]);
    }
  }
}

TEST(shuravina_o_jarvis_pass, Test_Empty_Points) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<Point> global_points = {};
  std::vector<Point> global_hull;

  JarvisPassMPI jarvis_mpi(global_points);
  ASSERT_FALSE(jarvis_mpi.validation());
  jarvis_mpi.run();
  global_hull = jarvis_mpi.get_hull();

  if (rank == 0) {
    std::vector<Point> seq_hull = jarvis_march(global_points);
    EXPECT_EQ(global_hull.size(), seq_hull.size());
  }
}

TEST(shuravina_o_jarvis_pass, Test_Cycle_Points) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<Point> global_points = {Point(0, 0), Point(2, 0), Point(2, 2), Point(0, 2), Point(1, 3)};
  std::vector<Point> global_hull;

  JarvisPassMPI jarvis_mpi(global_points);
  ASSERT_TRUE(jarvis_mpi.validation());
  jarvis_mpi.run();
  global_hull = jarvis_mpi.get_hull();

  if (rank == 0) {
    std::vector<Point> seq_hull = jarvis_march(global_points);
    ASSERT_EQ(global_hull.size(), seq_hull.size());
    for (size_t i = 0; i < global_hull.size(); i++) {
      EXPECT_EQ(global_hull[i], seq_hull[i]);
    }
  }
}

TEST(shuravina_o_jarvis_pass, Test_Rectangle_Points) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<Point> global_points = {Point(0, 0), Point(4, 0), Point(4, 3), Point(0, 3)};
  std::vector<Point> global_hull;

  JarvisPassMPI jarvis_mpi(global_points);
  ASSERT_TRUE(jarvis_mpi.validation());
  jarvis_mpi.run();
  global_hull = jarvis_mpi.get_hull();

  if (rank == 0) {
    std::vector<Point> seq_hull = jarvis_march(global_points);
    ASSERT_EQ(global_hull.size(), seq_hull.size());
    for (size_t i = 0; i < global_hull.size(); i++) {
      EXPECT_EQ(global_hull[i], seq_hull[i]);
    }
  }
}

TEST(shuravina_o_jarvis_pass, Test_Triangle_Points) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<Point> global_points = {Point(0, 0), Point(3, 0), Point(1, 4)};
  std::vector<Point> global_hull;

  JarvisPassMPI jarvis_mpi(global_points);
  ASSERT_TRUE(jarvis_mpi.validation());
  jarvis_mpi.run();
  global_hull = jarvis_mpi.get_hull();

  if (rank == 0) {
    std::vector<Point> seq_hull = jarvis_march(global_points);
    ASSERT_EQ(global_hull.size(), seq_hull.size());
    for (size_t i = 0; i < global_hull.size(); i++) {
      EXPECT_EQ(global_hull[i], seq_hull[i]);
    }
  }
}