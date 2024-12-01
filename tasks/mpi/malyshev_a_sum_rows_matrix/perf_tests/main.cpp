// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/malyshev_a_sum_rows_matrix/include/ops_mpi.hpp"

namespace malyshev_a_sum_rows_matrix_test_function {

std::vector<std::vector<int32_t>> getRandomData(uint32_t rows, uint32_t cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<std::vector<int32_t>> data(rows, std::vector<int32_t>(cols));

  for (auto &row : data) {
    for (auto &el : row) {
      el = -200 + gen() % (300 + 200 + 1);
    }
  }

  return data;
}

}  // namespace malyshev_a_sum_rows_matrix_test_function

TEST(malyshev_a_sum_rows_matrix_mpi, test_pipeline_run) {
  uint32_t rows = 3000;
  uint32_t cols = 3000;

  boost::mpi::communicator world;
  std::vector<std::vector<int32_t>> randomData;
  std::vector<int32_t> mpiSum;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_a_sum_rows_matrix_mpi::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomData = malyshev_a_sum_rows_matrix_test_function::getRandomData(rows, cols);
    mpiSum.resize(rows);

    for (auto &row : randomData) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiSum.data()));
    taskDataPar->outputs_count.push_back(rows);
  }

  auto testMpiTaskParallel = std::make_shared<malyshev_a_sum_rows_matrix_mpi::TestTaskParallel>(taskDataPar);

  ASSERT_TRUE(testMpiTaskParallel->validation());
  ASSERT_TRUE(testMpiTaskParallel->pre_processing());
  ASSERT_TRUE(testMpiTaskParallel->run());
  ASSERT_TRUE(testMpiTaskParallel->post_processing());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(malyshev_a_sum_rows_matrix_mpi, test_task_run) {
  uint32_t rows = 3000;
  uint32_t cols = 3000;

  boost::mpi::communicator world;
  std::vector<std::vector<int32_t>> randomData;
  std::vector<int32_t> mpiSum;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_a_sum_rows_matrix_mpi::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomData = malyshev_a_sum_rows_matrix_test_function::getRandomData(rows, cols);
    mpiSum.resize(rows);

    for (auto &row : randomData) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiSum.data()));
    taskDataPar->outputs_count.push_back(rows);
  }

  auto testMpiTaskParallel = std::make_shared<malyshev_a_sum_rows_matrix_mpi::TestTaskParallel>(taskDataPar);

  ASSERT_TRUE(testMpiTaskParallel->validation());
  ASSERT_TRUE(testMpiTaskParallel->pre_processing());
  ASSERT_TRUE(testMpiTaskParallel->run());
  ASSERT_TRUE(testMpiTaskParallel->post_processing());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) ppc::core::Perf::print_perf_statistic(perfResults);
}