// Copyright 2023 Nesterov Alexander
// drozhdinov_d_sum_cols_matrix func
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/drozhdinov_d_sum_cols_matrix/include/ops_mpi.hpp"

TEST(drozhdinov_d_sum_cols_matrix_mpi, ParallelTest1) {
  boost::mpi::communicator world;

  int cols = 2;
  int rows = 2;

  // Create data
  std::vector<int> matrix = {1, 0, 2, 1};
  std::vector<int> expres_par(cols, 0);
  std::vector<int> ans = {3, 1};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(expres_par.size());
  }

  drozhdinov_d_sum_cols_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> expres_seq(cols, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->inputs_count.emplace_back(rows);
    // taskDataSeq->inputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expres_seq.data()));
    taskDataSeq->outputs_count.emplace_back(expres_seq.size());

    // Create Task
    drozhdinov_d_sum_cols_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(expres_par, expres_seq);
  }
}

TEST(drozhdinov_d_sum_cols_matrix_mpi, ParallelTest2) {
  boost::mpi::communicator world;

  int cols = 2000;
  int rows = 2000;

  // Create data
  std::vector<int> matrix(cols * rows, 0);
  matrix[1] = 1;
  std::vector<int> expres_par(cols, 0);
  std::vector<int> ans(cols, 0);
  ans[1] = 1;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(expres_par.size());
  }

  drozhdinov_d_sum_cols_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> expres_seq(cols, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expres_seq.data()));
    taskDataSeq->outputs_count.emplace_back(expres_seq.size());

    // Create Task
    drozhdinov_d_sum_cols_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(expres_par, expres_seq);
  }
}

TEST(drozhdinov_d_sum_cols_matrix_mpi, ParallelTest3) {
  boost::mpi::communicator world;

  int cols = 10000;
  int rows = 10000;

  // Create data
  std::vector<int> matrix(cols * rows, 0);
  matrix[1] = 1;
  std::vector<int> expres_par(cols, 0);
  std::vector<int> ans(cols, 0);
  ans[1] = 1;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(expres_par.size());
  }

  drozhdinov_d_sum_cols_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> expres_seq(cols, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expres_seq.data()));
    taskDataSeq->outputs_count.emplace_back(expres_seq.size());

    // Create Task
    drozhdinov_d_sum_cols_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(expres_par, expres_seq);
  }
}