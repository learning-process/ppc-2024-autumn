// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <numbers>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/naumov_b_simpson_method/include/ops_mpi.hpp"

TEST(naumov_b_simpson_method_par, perf_pipeline_run) {
  auto func = [](double x) -> double { return std::sin(x) * std::log(x + 1.0); };
  double lower_bound = 0.0;
  double upper_bound = std::numbers::pi;
  int num_steps = 10000;
  double output = 0.0;

  boost::mpi::communicator world;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&lower_bound));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&upper_bound));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_steps));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&output));

  auto task = std::make_shared<naumov_b_simpson_method_mpi::TestMPITaskParallel>(task_data, func, lower_bound,
                                                                                 upper_bound, num_steps);

  auto perf_attributes = std::make_shared<ppc::core::PerfAttr>();
  perf_attributes->num_running = 10;

  auto start = std::chrono::high_resolution_clock::now();
  perf_attributes->current_timer = [&] {
    auto current = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current - start).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);

  perf_analyzer->pipeline_run(perf_attributes, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perf_results);
  }
}

TEST(naumov_b_simpson_method_par, perf_task_run) {
  auto func = [](double x) -> double { return std::exp(-x * x); };

  double lower_bound = 0.0;
  double upper_bound = 2.0;
  int num_steps = 10000;
  double output = 0.0;

  boost::mpi::communicator world;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&lower_bound));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&upper_bound));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_steps));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&output));

  auto task = std::make_shared<naumov_b_simpson_method_mpi::TestMPITaskParallel>(task_data, func, lower_bound,
                                                                                 upper_bound, num_steps);

  auto perf_attributes = std::make_shared<ppc::core::PerfAttr>();
  perf_attributes->num_running = 10;

  auto start = std::chrono::high_resolution_clock::now();
  perf_attributes->current_timer = [&] {
    auto current = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current - start).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);

  perf_analyzer->task_run(perf_attributes, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perf_results);
  }
}
