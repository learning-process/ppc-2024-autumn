// Copyright 2024 Korobeinikov Arseny
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/korobeinikov_dijkstras_algorithm/include/ops_mpi_korobeinikov.hpp"

TEST(mpi_korobeinikov_perf_test_lab_03, test_pipeline_run) {
  boost::mpi::communicator world;
  // Create data
  size_t size = 4000;

  std::vector<int> values(size * size, 1);
  std::vector<int> col(size * size, 1);
  std::vector<int> RowIndex(size + 1, 1);

  int sv = 0;
  std::vector<int> out(size, 0);
  std::vector<int> right_answer(size, 1);
  right_answer[0] = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (size_t i = 0; i < size; i++) {
      for (size_t j = 0; j < size; j++) {
        col[i * size + j] = j;
      }
    }
    for (size_t i = 0; i < size + 1; i++) {
      RowIndex[i] = i * size;
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(col.data()));
    taskDataPar->inputs_count.emplace_back(col.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(RowIndex.data()));
    taskDataPar->inputs_count.emplace_back(RowIndex.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&sv));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel = std::make_shared<korobeinikov_a_test_task_mpi_lab_03::TestMPITaskParallel>(taskDataPar);

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
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (size_t i = 0; i < right_answer.size(); i++) {
      ASSERT_EQ(right_answer[i], out[i]);
    }
  }
}

TEST(mpi_korobeinikov_perf_test_lab_03, test_task_run) {
  boost::mpi::communicator world;
  // Create data
  size_t size = 4000;

  std::vector<int> values(size * size, 1);
  std::vector<int> col(size * size, 1);
  std::vector<int> RowIndex(size + 1, 1);

  int sv = 0;
  std::vector<int> out(size, 0);
  std::vector<int> right_answer(size, 1);
  right_answer[0] = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (size_t i = 0; i < size; i++) {
      for (size_t j = 0; j < size; j++) {
        col[i * size + j] = j;
      }
    }
    for (size_t i = 0; i < size + 1; i++) {
      RowIndex[i] = i * size;
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(col.data()));
    taskDataPar->inputs_count.emplace_back(col.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(RowIndex.data()));
    taskDataPar->inputs_count.emplace_back(RowIndex.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&sv));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel = std::make_shared<korobeinikov_a_test_task_mpi_lab_03::TestMPITaskParallel>(taskDataPar);

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
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (size_t i = 0; i < right_answer.size(); i++) {
      ASSERT_EQ(right_answer[i], out[i]);
    }
  }
}