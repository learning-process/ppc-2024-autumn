// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/zolotareva_a_smoothing_image/include/ops_mpi.hpp"

TEST(mpi_zolotareva_a_smoothing_image_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;
  const uint32_t height = 100;
  const uint32_t width = 100;
  std::vector<uint8_t> input(width * height, 0);
  std::vector<uint8_t> output(width * height);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(input.data());
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->outputs.emplace_back(output.data());
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  auto testMpiTaskParallel = std::make_shared<zolotareva_a_smoothing_image_mpi::TestMPITaskParallel>(taskDataPar);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(height, taskDataPar->inputs_count.back());
  }
}

TEST(mpi_zolotareva_a_smoothing_image_perf_test, test_task_run) {
  boost::mpi::communicator world;
  const uint32_t height = 100;
  const uint32_t width = 100;
  std::vector<uint8_t> input(width * height, 0);
  std::vector<uint8_t> output(width * height);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(input.data());
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->outputs.emplace_back((output.data()));
    taskDataPar->outputs_count.emplace_back(output.size());
  }

  auto testMpiTaskParallel = std::make_shared<zolotareva_a_smoothing_image_mpi::TestMPITaskParallel>(taskDataPar);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(height, taskDataPar->inputs_count.back());
  }
}