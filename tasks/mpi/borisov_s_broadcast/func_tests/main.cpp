#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/borisov_s_broadcast/include/ops_mpi.hpp"

using namespace borisov_s_broadcast;

TEST(DistanceMatrixTask, ValidInput) {
  std::vector<double> points = {0.0, 0.0, 3.0, 4.0, 6.0, 8.0};

  size_t n = points.size() / 2;
  std::vector<double> expected_matrix = {0.0, 5.0, 10.0, 5.0, 0.0, 5.0, 10.0, 5.0, 0.0};

  auto taskDataPtr = std::make_shared<ppc::core::TaskData>();
  taskDataPtr->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataPtr->inputs_count.emplace_back(n);
  taskDataPtr->outputs.resize(1);
  taskDataPtr->outputs_count.emplace_back(n * n);
  taskDataPtr->outputs[0] = reinterpret_cast<uint8_t *>(new double[n * n]);

  auto task = std::make_shared<DistanceMatrixTaskSequential>(taskDataPtr);

  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  auto *output = reinterpret_cast<double *>(taskDataPtr->outputs[0]);
  for (size_t i = 0; i < n * n; i++) {
    EXPECT_NEAR(output[i], expected_matrix[i], 1e-6);
  }

  delete[] output;
}

TEST(DistanceMatrixTask, EmptyInput) {
  auto taskDataPtr = std::make_shared<ppc::core::TaskData>();

  taskDataPtr->inputs_count.emplace_back(0);
  taskDataPtr->outputs_count.emplace_back(0);
  taskDataPtr->outputs.emplace_back(nullptr);
  taskDataPtr->inputs.emplace_back(nullptr);

  auto task = std::make_shared<DistanceMatrixTaskSequential>(taskDataPtr);

  ASSERT_FALSE(task->validation());
}

TEST(DistanceMatrixTask, MissingOutputBuffer) {
  std::vector<double> points = {0.0, 0.0, 3.0, 4.0};

  auto taskDataPtr = std::make_shared<ppc::core::TaskData>();
  taskDataPtr->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataPtr->inputs_count.emplace_back(2);
  taskDataPtr->outputs_count.emplace_back(0);
  taskDataPtr->outputs.emplace_back(nullptr);

  auto task = std::make_shared<DistanceMatrixTaskSequential>(taskDataPtr);

  ASSERT_FALSE(task->validation());
}

TEST(DistanceMatrixTask, SinglePoint) {
  std::vector<double> points = {1.0, 2.0};

  auto taskDataPtr = std::make_shared<ppc::core::TaskData>();
  taskDataPtr->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataPtr->inputs_count.emplace_back(1);
  taskDataPtr->outputs.resize(1);
  taskDataPtr->outputs_count.emplace_back(1);
  taskDataPtr->outputs[0] = reinterpret_cast<uint8_t *>(new double[1]);

  auto task = std::make_shared<DistanceMatrixTaskSequential>(taskDataPtr);

  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  auto *output = reinterpret_cast<double *>(taskDataPtr->outputs[0]);
  EXPECT_NEAR(output[0], 0.0, 1e-6);

  delete[] output;
}

TEST(DistanceMatrixTask, LargeInput) {
  const int point_count = 1000;
  auto points = getRandomPoints(point_count);

  auto taskDataPtr = std::make_shared<ppc::core::TaskData>();
  taskDataPtr->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataPtr->inputs_count.emplace_back(point_count);
  taskDataPtr->outputs.resize(1);
  taskDataPtr->outputs_count.emplace_back(point_count * point_count);
  taskDataPtr->outputs[0] = reinterpret_cast<uint8_t *>(new double[point_count * point_count]);

  auto task = std::make_shared<DistanceMatrixTaskSequential>(taskDataPtr);

  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  auto *output = reinterpret_cast<double *>(taskDataPtr->outputs[0]);
  for (size_t i = 0; i < point_count; i++) {
    EXPECT_GE(output[(i * point_count) + i], 0.0);
  }

  delete[] output;
}

TEST(Parallel_Operations_MPI, Test_DistanceMatrix) {
  boost::mpi::communicator world;
  std::vector<double> global_points;
  std::vector<double> global_distance_matrix;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  size_t num_points = 0;

  if (world.rank() == 0) {
    num_points = 3;
    global_points = {0.0, 0.0, 3.0, 4.0, 6.0, 8.0};

    global_distance_matrix.resize(num_points * num_points, 0.0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_points.data()));
    taskDataPar->inputs_count.emplace_back(num_points);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_distance_matrix.data()));
    taskDataPar->outputs_count.emplace_back(global_distance_matrix.size());
  }

  boost::mpi::broadcast(world, num_points, 0);

  if (world.rank() != 0) {
    global_points.resize(num_points * 2);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_points.data()));
    taskDataPar->inputs_count.emplace_back(num_points);

    global_distance_matrix.resize(num_points * num_points, 0.0);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_distance_matrix.data()));
    taskDataPar->outputs_count.emplace_back(global_distance_matrix.size());
  }

  borisov_s_broadcast::DistanceMatrixTaskParallel parallelTask(taskDataPar);
  ASSERT_TRUE(parallelTask.validation());
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    const std::vector<double> reference_distance_matrix = {0.0, 5.0, 10.0, 5.0, 0.0, 5.0, 10.0, 5.0, 0.0};

    ASSERT_EQ(global_distance_matrix.size(), reference_distance_matrix.size());
    for (size_t i = 0; i < global_distance_matrix.size(); i++) {
      ASSERT_NEAR(global_distance_matrix[i], reference_distance_matrix[i], 1e-2);
    }
  }
}

TEST(Parallel_Operations_MPI, Test_EmptyInput) {
  boost::mpi::communicator world;
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->outputs.emplace_back(nullptr);
  taskDataPar->inputs.emplace_back(nullptr);
  taskDataPar->inputs_count.emplace_back(0);
  taskDataPar->outputs_count.emplace_back(0);

  borisov_s_broadcast::DistanceMatrixTaskParallel parallelTask(taskDataPar);
  ASSERT_FALSE(parallelTask.validation());
}

TEST(Parallel_Operations_MPI, Test_SinglePoint) {
  boost::mpi::communicator world;
  std::vector<double> global_points;
  std::vector<double> global_distance_matrix;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  size_t num_points = 0;

  if (world.rank() == 0) {
    num_points = 1;
    global_points = {5.0, 5.0};
    global_distance_matrix.resize(1, 0.0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_points.data()));
    taskDataPar->inputs_count.emplace_back(num_points);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_distance_matrix.data()));
    taskDataPar->outputs_count.emplace_back(global_distance_matrix.size());
  }

  boost::mpi::broadcast(world, num_points, 0);

  if (world.rank() != 0) {
    global_points.resize(num_points * 2);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_points.data()));
    taskDataPar->inputs_count.emplace_back(num_points);

    global_distance_matrix.resize(num_points * num_points, 0.0);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_distance_matrix.data()));
    taskDataPar->outputs_count.emplace_back(global_distance_matrix.size());
  }

  borisov_s_broadcast::DistanceMatrixTaskParallel parallelTask(taskDataPar);
  ASSERT_TRUE(parallelTask.validation());
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_distance_matrix.size(), static_cast<size_t>(1));
    ASSERT_NEAR(global_distance_matrix[0], 0.0, 1e-6);
  }
}
