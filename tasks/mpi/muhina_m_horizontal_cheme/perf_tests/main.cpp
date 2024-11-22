// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/muhina_m_horizontal_cheme/include/ops_mpi.hpp"

using namespace muhina_m_horizontal_cheme_mpi;

TEST(muhina_m_horizontal_cheme_mpi, run_pipeline) {
  boost::mpi::communicator world;

  std::vector<int> matrix;
  std::vector<int> vector;
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int num_rows;
  int num_cols;

  if (world.rank() == 0) {
    num_rows = 1024;
    num_cols = 1024;

    matrix.resize(num_rows * num_cols);
    for (int j = 0; j < num_rows; ++j) {
      for (int i = 0; i < num_cols; ++i) {
        matrix[j * num_cols + i] = rand() % 100;
      }
    }

    vector.resize(num_rows);
    for (int i = 0; i < num_rows; ++i) {
      vector[i] = rand() % 100;
    }

    result.resize(num_cols, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
    taskDataPar->inputs_count.emplace_back(vector.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }

  auto taskParallel = std::make_shared<HorizontalSchemeMPIParallel>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    std::vector<int> seq_result(result.size(), 0);

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
    taskDataSeq->inputs_count.emplace_back(vector.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
    taskDataSeq->outputs_count.emplace_back(seq_result.size());

    auto taskSequential = std::make_shared<HorizontalSchemeMPISequential>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();

    ASSERT_EQ(result.size(), seq_result.size());
    for (size_t i = 0; i < result.size(); ++i) {
      ASSERT_EQ(result[i], seq_result[i]);
    }
  }
}

TEST(muhina_m_horizontal_cheme_mpi, run_task) {
  boost::mpi::communicator world;

  std::vector<int> matrix;
  std::vector<int> vector;
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int num_rows;
  int num_cols;

  if (world.rank() == 0) {
    num_rows = 1000;
    num_cols = 1000;

    matrix.resize(num_rows * num_cols);
    for (int j = 0; j < num_rows; ++j) {
      for (int i = 0; i < num_cols; ++i) {
        matrix[j * num_cols + i] = rand() % 100;
      }
    }

    vector.resize(num_rows);
    for (int i = 0; i < num_rows; ++i) {
      vector[i] = rand() % 100;
    }

    result.resize(num_cols, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
    taskDataPar->inputs_count.emplace_back(vector.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }

  auto taskParallel = std::make_shared<HorizontalSchemeMPIParallel>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    std::vector<int> seq_result(result.size(), 0);

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
    taskDataSeq->inputs_count.emplace_back(vector.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
    taskDataSeq->outputs_count.emplace_back(seq_result.size());

    auto taskSequential = std::make_shared<HorizontalSchemeMPISequential>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();

    ASSERT_EQ(result.size(), seq_result.size());
    for (size_t i = 0; i < result.size(); ++i) {
      ASSERT_EQ(result[i], seq_result[i]);
    }
  }
}
