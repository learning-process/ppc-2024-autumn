// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/gordeva_t_max_val_of_column_matrix/include/ops_mpi.hpp"

TEST(gordeva_t_max_val_of_column_matrix_mpi, IsEmptyInput) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) ASSERT_FALSE(testMpiTaskParallel.validation());
}

TEST(gordeva_t_max_val_of_column_matrix_mpi, IsEmptyOutput) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    taskDataPar->inputs_count.push_back(5);
    taskDataPar->inputs_count.push_back(5);
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(new int[25]));
    ASSERT_FALSE(testMpiTaskParallel.validation());

    delete[] reinterpret_cast<int*>(taskDataPar->inputs[0]);
  }
}

TEST(gordeva_t_max_val_of_column_matrix_mpi, Max_val_of_5000_columns_with_random) {
  boost::mpi::communicator world;

  // Create data
  const int rows = 5000;
  const int cols = 5000;
  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> global_max(cols, INT_MIN);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matr = gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential::gen_rand_matr(rows, cols);
    for (unsigned int i = 0; i < global_matr.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matr[i].data()));
    }
    taskDataPar->inputs_count = {rows, cols};
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  // Create Task
  gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> max_example(cols, INT_MIN);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matr.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matr[i].data()));
    }
    taskDataSeq->inputs_count = {rows, cols};
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(max_example.data()));
    taskDataSeq->outputs_count.emplace_back(max_example.size());
    gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (int j = 0; j < cols; j++) {
      ASSERT_EQ(global_max[j], 200);
    }
  }
}

TEST(gordeva_t_max_val_of_column_matrix_mpi, Max_val_of_5000_7000_columns_with_random) {
  boost::mpi::communicator world;

  // Create data
  const int rows = 5000;
  const int cols = 7000;
  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> global_max(cols, INT_MIN);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matr = gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential::gen_rand_matr(rows, cols);
    for (unsigned int i = 0; i < global_matr.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matr[i].data()));
    }
    taskDataPar->inputs_count = {rows, cols};
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  // Create Task
  gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> max_example(cols, INT_MIN);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matr.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matr[i].data()));
    }
    taskDataSeq->inputs_count = {rows, cols};
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(max_example.data()));
    taskDataSeq->outputs_count.emplace_back(max_example.size());
    gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (int j = 0; j < cols; j++) {
      ASSERT_EQ(global_max[j], 200);
    }
  }
}

TEST(gordeva_t_max_val_of_column_matrix_mpi, Max_val_of_5000_10000_columns_with_random) {
  boost::mpi::communicator world;

  // Create data
  const int rows = 5000;
  const int cols = 10000;
  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> global_max(cols, INT_MIN);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
      global_matr = gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential::gen_rand_matr(rows,cols);
    for (unsigned int i = 0; i < global_matr.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matr[i].data()));
    }
    taskDataPar->inputs_count = {rows, cols};
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  // Create Task
  gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> max_example(cols, INT_MIN);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matr.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matr[i].data()));
    }
    taskDataSeq->inputs_count = {rows, cols};
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(max_example.data()));
    taskDataSeq->outputs_count.emplace_back(max_example.size());
    gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (int j = 0; j < cols; j++) {
      ASSERT_EQ(global_max[j], 200);
    }
  }
}
