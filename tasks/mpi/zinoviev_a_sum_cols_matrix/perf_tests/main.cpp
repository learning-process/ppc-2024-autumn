// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/zinoviev_a_sum_cols_matrix/include/ops_mpi.hpp"

TEST(zinoviev_a_sum_cols_matrix, test_task_run) {
  boost::mpi::communicator world;
  int cols = 10;
  int rows = 10;

  std::vector<int> matrix(cols * rows, 0);
  matrix[3] = 1;
  std::vector<int> expres(cols, 0);
  std::vector<int> ans(cols, 0);
  ans[3] = 1;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(expres.data()));
    taskDataPar->outputs_count.emplace_back(expres.size());
  }

  auto testMpiTaskParallel = std::make_shared<zinoviev_a_sum_cols_matrix_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(expres, ans);
  }
}

TEST(zinoviev_a_sum_cols_matrix, test_pipeline_run) {
  boost::mpi::communicator world;

  int cols = 50;
  int rows = 1000;

  std::vector<int> matrix(cols * rows, 1);
  std::vector<int> expres(cols, 1000);
  std::vector<int> ans(cols, 1000);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(expres.data()));
    taskDataPar->outputs_count.emplace_back(expres.size());
  }

  auto testMpiTaskParallel = std::make_shared<zinoviev_a_sum_cols_matrix_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(expres, ans);
  }
}