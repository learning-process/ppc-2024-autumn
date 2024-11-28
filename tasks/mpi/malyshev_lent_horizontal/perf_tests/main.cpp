// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/malyshev_lent_horizontal/include/ops_mpi.hpp"

namespace malyshev_lent_horizontal {

std::vector<std::vector<int32_t>> generateRandomMatrix(uint32_t rows, uint32_t cols) {
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

std::vector<int32_t> generateRandomVector(uint32_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int32_t> data(size);

  for (auto &el : data) {
    el = -200 + gen() % (300 + 200 + 1);
  }

  return data;
}

}  // namespace malyshev_lent_horizontal

TEST(malyshev_lent_horizontal, test_pipeline_run) {
  uint32_t rows = 3000;
  uint32_t cols = 3000;

  boost::mpi::communicator world;
  std::vector<std::vector<int32_t>> randomMatrix;
  std::vector<int32_t> randomVector;
  std::vector<int32_t> mpiResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_lent_horizontal::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomMatrix = malyshev_lent_horizontal::generateRandomMatrix(rows, cols);
    randomVector = malyshev_lent_horizontal::generateRandomVector(cols);
    mpiResult.resize(rows);

    for (auto &row : randomMatrix) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiResult.data()));
    taskDataPar->outputs_count.push_back(rows);
  }

  auto testMpiTaskParallel = std::make_shared<malyshev_lent_horizontal::TestTaskParallel>(taskDataPar);

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

TEST(malyshev_lent_horizontal, test_task_run) {
  uint32_t rows = 3000;
  uint32_t cols = 3000;

  boost::mpi::communicator world;
  std::vector<std::vector<int32_t>> randomMatrix;
  std::vector<int32_t> randomVector;
  std::vector<int32_t> mpiResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_lent_horizontal::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomMatrix = malyshev_lent_horizontal::generateRandomMatrix(rows, cols);
    randomVector = malyshev_lent_horizontal::generateRandomVector(cols);
    mpiResult.resize(rows);

    for (auto &row : randomMatrix) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataPar->inputs_count.push_back(rows);
    taskDataPar->inputs_count.push_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiResult.data()));
    taskDataPar->outputs_count.push_back(rows);
  }

  auto testMpiTaskParallel = std::make_shared<malyshev_lent_horizontal::TestTaskParallel>(taskDataPar);

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