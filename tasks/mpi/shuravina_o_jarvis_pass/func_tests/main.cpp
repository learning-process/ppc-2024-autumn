#include <gtest/gtest.h>
#include <mpi.h>

#include <vector>

#include "mpi/shuravina_o_jarvis_pass/include/ops_mpi.hpp"

using namespace shuravina_o_jarvis_pass;

bool mpi_initialized = false;

void MPI_Setup() {
  if (!mpi_initialized) {
    MPI_Init(nullptr, nullptr);
    mpi_initialized = true;
  }
}

void MPI_Teardown() {
  if (mpi_initialized) {
    MPI_Finalize();
  }
}

class MPITestEnvironment : public ::testing::Environment {
 public:
  virtual void SetUp() override { MPI_Setup(); }

  virtual void TearDown() override { MPI_Teardown(); }
};

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  ::testing::AddGlobalTestEnvironment(new MPITestEnvironment);

  return RUN_ALL_TESTS();
}

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

TEST(shuravina_o_jarvis_pass, Test_Hull_Size) {
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

    EXPECT_EQ(global_hull.size(), seq_hull.size());
  }
}

TEST(shuravina_o_jarvis_pass, Test_Validation) {
  std::vector<Point> valid_points = {Point(0, 0), Point(1, 1), Point(0, 2)};
  std::vector<Point> invalid_points = {Point(0, 0)};

  JarvisPassMPI jarvis_mpi_valid(valid_points);
  JarvisPassMPI jarvis_mpi_invalid(invalid_points);

  EXPECT_TRUE(jarvis_mpi_valid.validation());
  EXPECT_FALSE(jarvis_mpi_invalid.validation());
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