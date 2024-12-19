// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <cmath>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kalyakina_a_trapezoidal_integration_mpi/include/ops_mpi.hpp"

double function(std::vector<double> input) { return (-3 * pow(input[1], 2) * sin(5 * input[0])) / 2; };

TEST(kalyakina_a_trapezoidal_integration_mpi, TrapezoidalIntegrationParallel_pipeline_run) {
  boost::mpi::communicator world;

  std::vector<std::pair<double, double>> limits;
  std::vector<unsigned int> intervals;
  std::vector<unsigned int> count;
  std::vector<double> out(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    limits = {{0.0, 1.0}, {4.0, 6.0}};
    intervals = {500, 500};
    count = std::vector<unsigned int>{2};
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(count.data()));
    taskDataParallel->inputs_count.emplace_back(count.size());
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(limits.data()));
    taskDataParallel->inputs_count.emplace_back(limits.size());
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(intervals.data()));
    taskDataParallel->inputs_count.emplace_back(intervals.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataParallel->outputs_count.emplace_back(out.size());
  }

  auto testTaskParallel = std::make_shared<kalyakina_a_trapezoidal_integration_mpi::TrapezoidalIntegrationTaskParallel>(
      taskDataParallel, function);
  ASSERT_EQ(testTaskParallel->validation(), true);
  testTaskParallel->pre_processing();
  testTaskParallel->run();
  testTaskParallel->post_processing();

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
    ASSERT_NEAR((pow(6.0, 3) - pow(4.0, 3)) * (cos(5 * 1.0) - cos(5 * 0.0)) / 10, out[0], 0.001);
  }
}

TEST(kalyakina_a_trapezoidal_integration_mpi, TrapezoidalIntegrationParallel_task_run) {
  boost::mpi::communicator world;

  std::vector<std::pair<double, double>> limits;
  std::vector<unsigned int> intervals;
  std::vector<unsigned int> count;
  std::vector<double> out(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    limits = {{0.0, 1.0}, {4.0, 6.0}};
    intervals = {500, 500};
    count = std::vector<unsigned int>{2};
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(count.data()));
    taskDataParallel->inputs_count.emplace_back(count.size());
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(limits.data()));
    taskDataParallel->inputs_count.emplace_back(limits.size());
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(intervals.data()));
    taskDataParallel->inputs_count.emplace_back(intervals.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataParallel->outputs_count.emplace_back(out.size());
  }

  auto testTaskParallel = std::make_shared<kalyakina_a_trapezoidal_integration_mpi::TrapezoidalIntegrationTaskParallel>(
      taskDataParallel, function);
  ASSERT_EQ(testTaskParallel->validation(), true);
  testTaskParallel->pre_processing();
  testTaskParallel->run();
  testTaskParallel->post_processing();

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
    ASSERT_NEAR((pow(6.0, 3) - pow(4.0, 3)) * (cos(5 * 1.0) - cos(5 * 0.0)) / 10, out[0], 0.001);
  }
}