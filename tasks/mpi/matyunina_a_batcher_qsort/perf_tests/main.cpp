// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/matyunina_a_batcher_qsort/include/ops_mpi.hpp"

TEST(matyunina_a_batcher_qsort_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int32_t> in;
  std::vector<int32_t> out;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    uint32_t size = 25000;
    in.resize(size);
    out.resize(size);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in.size());

    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  // Create Task
  auto testTaskParallel = std::make_shared<matyunina_a_batcher_qsort_mpi::TestTaskParallel<int32_t>>(taskData);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(matyunina_a_batcher_qsort_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int32_t> in;
  std::vector<int32_t> out;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    uint32_t size = 25000;
    in.resize(size);
    out.resize(size);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in.size());

    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  // Create Task
  auto testTaskParallel = std::make_shared<matyunina_a_batcher_qsort_mpi::TestTaskParallel<int32_t>>(taskData);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}
