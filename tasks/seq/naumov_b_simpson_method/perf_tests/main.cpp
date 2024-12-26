// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <numbers>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/naumov_b_simpson_method/include/ops_seq.hpp"

TEST(naumov_b_simpson_method_seq_perf_pipeline_run, perf_pipeline_run) {
  auto func = [](double x) -> double { return std::sin(x) * std::log(x + 1.0); };

  naumov_b_simpson_method_seq::bound_t bounds = {0.0, std::numbers::pi};
  int num_steps = 100;
  double output = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&func));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bounds));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_steps));
  task_data->inputs_count.emplace_back(1);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&output));
  task_data->outputs_count.emplace_back(1);

  auto task = std::make_shared<naumov_b_simpson_method_seq::TestTaskSequential>(task_data, func, bounds, num_steps);

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

  ppc::core::Perf::print_perf_statistic(perf_results);
}

TEST(naumov_b_simpson_method_seq_perf_task_run, perf_task_run) {
  auto func = [](double x) -> double { return std::exp(-x * x); };

  naumov_b_simpson_method_seq::bound_t bounds = {0.0, 2.0};
  int num_steps = 100;
  double output = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&func));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bounds));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_steps));
  task_data->inputs_count.emplace_back(1);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&output));
  task_data->outputs_count.emplace_back(1);

  auto task = std::make_shared<naumov_b_simpson_method_seq::TestTaskSequential>(task_data, func, bounds, num_steps);

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

  ppc::core::Perf::print_perf_statistic(perf_results);
}
