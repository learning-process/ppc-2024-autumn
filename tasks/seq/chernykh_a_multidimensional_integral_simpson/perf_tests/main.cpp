#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <numbers>

#include "core/perf/include/perf.hpp"
#include "seq/chernykh_a_multidimensional_integral_simpson/include/ops_seq.hpp"

namespace chernykh_a_multidimensional_integral_simpson_seq {

enum class RunType : uint8_t { TASK, PIPELINE };

void run_task(RunType run_type, func_nd_t func, bounds_t &bounds, step_range_t &step_range, double tolerance,
              double want) {
  auto output = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&func));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(bounds.data()));
  task_data->inputs_count.emplace_back(bounds.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&step_range));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&tolerance));
  task_data->inputs_count.emplace_back(1);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&output));
  task_data->outputs_count.emplace_back(1);

  auto task = std::make_shared<SequentialTask>(task_data);

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

  switch (run_type) {
    case RunType::PIPELINE:
      perf_analyzer->pipeline_run(perf_attributes, perf_results);
      break;
    case RunType::TASK:
      perf_analyzer->task_run(perf_attributes, perf_results);
      break;
  }

  ppc::core::Perf::print_perf_statistic(perf_results);
  EXPECT_NEAR(want, output, tolerance);
}

}  // namespace chernykh_a_multidimensional_integral_simpson_seq

namespace chernykh_a_mis_seq = chernykh_a_multidimensional_integral_simpson_seq;

TEST(chernykh_a_multidimensional_integral_simpson_seq, test_pipeline_run) {
  auto func = [](const chernykh_a_mis_seq::func_args_t &args) -> double {
    return std::exp(-args[0] - args[1] - args[2]) * std::sin(args[0]) * std::sin(args[1]) * std::sin(args[2]);
  };
  chernykh_a_mis_seq::bounds_t bounds = {{0.0, std::numbers::pi}, {0.0, std::numbers::pi}, {0.0, std::numbers::pi}};
  chernykh_a_mis_seq::step_range_t step_range = {2, 1000};
  double tolerance = 1e-6;
  double want = std::pow((1.0 + std::exp(-std::numbers::pi)) / 2.0, 3);
  chernykh_a_mis_seq::run_task(chernykh_a_mis_seq::RunType::PIPELINE, func, bounds, step_range, tolerance, want);
}

TEST(chernykh_a_multidimensional_integral_simpson_seq, test_task_run) {
  auto func = [](const chernykh_a_mis_seq::func_args_t &args) -> double {
    return std::exp(-args[0] - args[1] - args[2]) * std::sin(args[0]) * std::sin(args[1]) * std::sin(args[2]);
  };
  chernykh_a_mis_seq::bounds_t bounds = {{0.0, std::numbers::pi}, {0.0, std::numbers::pi}, {0.0, std::numbers::pi}};
  chernykh_a_mis_seq::step_range_t step_range = {2, 1000};
  double tolerance = 1e-6;
  double want = std::pow((1.0 + std::exp(-std::numbers::pi)) / 2.0, 3);
  chernykh_a_mis_seq::run_task(chernykh_a_mis_seq::RunType::TASK, func, bounds, step_range, tolerance, want);
}
