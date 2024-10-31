// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/kondratev_ya_max_col_matrix/include/ops_mpi.hpp"

void runTask(ppc::core::Task& task) {
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
}

void fillTaskData(std::shared_ptr<ppc::core::TaskData>& taskData, uint32_t row, uint32_t col, auto& mtrx, auto& res) {
  for (auto& mtrxRow : mtrx) taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(mtrxRow.data()));
  taskData->inputs_count.emplace_back(row);
  taskData->inputs_count.emplace_back(col);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  taskData->outputs_count.emplace_back(res.size());
}

TEST(kondratev_ya_max_col_matrix_mpi, test_1) {
  uint32_t row = 100;
  uint32_t col = 100;

  boost::mpi::communicator world;
  std::vector<int32_t> res(col);
  std::vector<std::vector<int32_t>> mtrx;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    mtrx = kondratev_ya_max_col_matrix_mpi::getRandomMatrix(row, col);
    fillTaskData(taskDataPar, row, col, mtrx, res);
  }

  kondratev_ya_max_col_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  runTask(testMpiTaskParallel);

  if (world.rank() == 0) {
    std::vector<int32_t> ref(col);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    fillTaskData(taskDataSeq, row, col, mtrx, ref);

    kondratev_ya_max_col_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    runTask(testMpiTaskSequential);

    for (uint32_t i = 0; i < res.size(); i++) ASSERT_EQ(res[i], ref[i]);
  }
}

TEST(kondratev_ya_max_col_matrix_mpi, test_2) {
  uint32_t row = 1000;
  uint32_t col = 50;

  boost::mpi::communicator world;
  std::vector<int32_t> res(col);
  std::vector<std::vector<int32_t>> mtrx;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    mtrx = kondratev_ya_max_col_matrix_mpi::getRandomMatrix(row, col);
    fillTaskData(taskDataPar, row, col, mtrx, res);
  }

  kondratev_ya_max_col_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  runTask(testMpiTaskParallel);
  if (world.rank() == 0) {
    std::vector<int32_t> ref(col);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    fillTaskData(taskDataSeq, row, col, mtrx, ref);

    kondratev_ya_max_col_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    runTask(testMpiTaskSequential);

    for (uint32_t i = 0; i < res.size(); i++) ASSERT_EQ(res[i], ref[i]);
  }
}

TEST(kondratev_ya_max_col_matrix_mpi, test_3) {
  uint32_t row = 500;
  uint32_t col = 1000;

  boost::mpi::communicator world;
  std::vector<int32_t> res(col);
  std::vector<std::vector<int32_t>> mtrx;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    mtrx = kondratev_ya_max_col_matrix_mpi::getRandomMatrix(row, col);
    fillTaskData(taskDataPar, row, col, mtrx, res);
  }

  kondratev_ya_max_col_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  runTask(testMpiTaskParallel);

  if (world.rank() == 0) {
    std::vector<int32_t> ref(col);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    fillTaskData(taskDataSeq, row, col, mtrx, ref);

    kondratev_ya_max_col_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    runTask(testMpiTaskSequential);

    for (uint32_t i = 0; i < res.size(); i++) ASSERT_EQ(res[i], ref[i]);
  }
}

TEST(kondratev_ya_max_col_matrix_mpi, right_insert_ref) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    uint32_t row = 3;
    uint32_t col = 3;
    int32_t ref = INT_MAX;

    auto mtrx = kondratev_ya_max_col_matrix_mpi::getRandomMatrix(row, col);
    kondratev_ya_max_col_matrix_mpi::insertRefValue(mtrx, ref);

    bool flag;
    for (uint32_t j = 0; j < col; j++) {
      flag = false;
      for (uint32_t i = 0; i < row; i++) {
        if (mtrx[i][j] == ref) {
          flag = true;
          break;
        }
      }

      ASSERT_TRUE(flag);
    }
  }
}
