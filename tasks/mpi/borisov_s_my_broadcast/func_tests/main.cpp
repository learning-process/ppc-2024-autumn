#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <vector>

#include "mpi/borisov_s_my_broadcast/include/ops_mpi.hpp"

using namespace borisov_s_my_broadcast;

TEST(Parallel_Operations_MPI2, Test_Fixed_Example_1) {
  boost::mpi::communicator world;

  size_t rows = 2;
  size_t cols = 2;
  double epsilon = 1.5;

  std::vector<double> global_matrix = {3.0, 4.0, 6.0, 8.0};
  std::vector<int> global_result(rows, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
    taskDataPar->outputs_count.push_back(global_result.size());
  }

  MPITaskMatrixClustering task(taskDataPar);

  if (world.rank() == 0) {
    ASSERT_TRUE(task.validation());
  } else {
    task.validation();
  }

  task.pre_processing();
  task.run();
  task.post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected_result = {1, 1};
    ASSERT_EQ(global_result, expected_result);
  }
}

TEST(Parallel_Operations_MPI2, Test_Fixed_Example_3x2) {
  boost::mpi::communicator world;

  size_t rows = 3;
  size_t cols = 2;
  double epsilon = 0.05;

  std::vector<double> global_matrix = {3.0, 4.0, 6.0, 8.0, 1.0, 1.0};
  std::vector<int> global_result(rows, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
    taskDataPar->outputs_count.push_back(global_result.size());
  }

  MPITaskMatrixClustering task(taskDataPar);

  if (world.rank() == 0) {
    ASSERT_TRUE(task.validation());
  } else {
    task.validation();
  }

  task.pre_processing();
  task.run();
  task.post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected_result = {1, 1, 0};
    ASSERT_EQ(global_result, expected_result);
  }
}

TEST(Parallel_Operations_MPI2, Test_Fixed_Example_2x3) {
  boost::mpi::communicator world;

  size_t rows = 2;
  size_t cols = 3;
  double epsilon = 1.0;

  std::vector<double> global_matrix = {3.0, 4.0, 5.0, 7.0, 8.0, 9.0};
  std::vector<int> global_result(rows, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
    taskDataPar->outputs_count.push_back(global_result.size());
  }

  MPITaskMatrixClustering task(taskDataPar);

  if (world.rank() == 0) {
    ASSERT_TRUE(task.validation());
  } else {
    task.validation();
  }

  task.pre_processing();
  task.run();
  task.post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected_result = {1, 1};
    ASSERT_EQ(global_result, expected_result);
  }
}

TEST(Parallel_Operations_MPI2, Validation_InsufficientInputs) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.clear();
    taskData->inputs_count.clear();
  }

  MPITaskMatrixClustering task(taskData);

  if (world.rank() == 0) {
    ASSERT_FALSE(task.validation());
  }
}

TEST(Parallel_Operations_MPI2, Validation_InvalidMatrixSize) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    size_t rows = 0;
    size_t cols = 3;
    double epsilon = 1.0;

    std::vector<double> global_matrix = {};

    taskData->inputs.clear();
    taskData->inputs_count.clear();

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskData->inputs_count.push_back(rows);
    taskData->inputs_count.push_back(cols);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  }

  MPITaskMatrixClustering task(taskData);

  if (world.rank() == 0) {
    ASSERT_FALSE(task.validation());
  }
}