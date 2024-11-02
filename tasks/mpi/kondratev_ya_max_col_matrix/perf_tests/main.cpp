// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kondratev_ya_max_col_matrix/include/ops_mpi.hpp"

std::vector<std::vector<int32_t>> getRandomMatrix(uint32_t row, uint32_t col) {
  int32_t low = -200;
  int32_t high = 200;

  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<std::vector<int32_t>> mtrx(row, std::vector<int32_t>(col));
  for (uint32_t i = 0; i < row; i++) {
    for (uint32_t j = 0; j < col; j++) {
      mtrx[i][j] = low + gen() % (high - low + 1);
    }
  }
  return mtrx;
}

void insertRefValue(std::vector<std::vector<int32_t>>& mtrx, int32_t ref) {
  std::random_device dev;
  std::mt19937 gen(dev());

  uint32_t ind;
  uint32_t row = mtrx.size();
  uint32_t col = mtrx[0].size();

  for (uint32_t j = 0; j < col; j++) {
    ind = gen() % row;
    mtrx[ind][j] = ref;
  }
}

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

TEST(kondratev_ya_max_col_matrix_mpi, test_pipeline_run) {
  uint32_t row = 6000;
  uint32_t col = 6000;
  int32_t ref_val = INT_MAX;

  boost::mpi::communicator world;
  std::vector<int32_t> res(col);
  std::vector<int32_t> ref(col, ref_val);
  std::vector<std::vector<int32_t>> mtrx;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    mtrx = getRandomMatrix(row, col);
    insertRefValue(mtrx, ref_val);
    fillTaskData(taskDataPar, row, col, mtrx, res);
  }

  auto testMpiTaskParallel = std::make_shared<kondratev_ya_max_col_matrix_mpi::TestMPITaskParallel>(taskDataPar);
  runTask(*testMpiTaskParallel);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  for (uint32_t i = 0; i < res.size(); i++) ASSERT_EQ(res[i], ref[i]);
}

TEST(kondratev_ya_max_col_matrix_mpi_perf_test, test_task_run) {
  uint32_t row = 6000;
  uint32_t col = 6000;
  int32_t ref_val = INT_MAX;

  boost::mpi::communicator world;
  std::vector<int32_t> res(col);
  std::vector<int32_t> ref(col, ref_val);
  std::vector<std::vector<int32_t>> mtrx;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    mtrx = getRandomMatrix(row, col);
    insertRefValue(mtrx, ref_val);
    fillTaskData(taskDataPar, row, col, mtrx, res);
  }

  auto testMpiTaskParallel = std::make_shared<kondratev_ya_max_col_matrix_mpi::TestMPITaskParallel>(taskDataPar);
  runTask(*testMpiTaskParallel);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  for (uint32_t i = 0; i < res.size(); i++) ASSERT_EQ(res[i], ref[i]);
}
