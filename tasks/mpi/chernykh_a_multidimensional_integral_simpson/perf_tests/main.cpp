#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <cmath>
#include <functional>
#include <memory>
#include <numbers>

#include "core/perf/include/perf.hpp"
#include "mpi/chernykh_a_multidimensional_integral_simpson/include/ops_mpi.hpp"

namespace chernykh_a_multidimensional_integral_simpson_mpi {

enum class RunType : uint8_t { TASK, PIPELINE };

void run_task(RunType run_type, func_nd_t func, bounds_t &bounds, step_range_t &step_range, double tolerance,
              double want) {
  auto world = boost::mpi::communicator();

  auto par_output = 0.0;
  auto par_task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    par_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(bounds.data()));
    par_task_data->inputs_count.emplace_back(bounds.size());
    par_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&step_range));
    par_task_data->inputs_count.emplace_back(1);
    par_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&tolerance));
    par_task_data->inputs_count.emplace_back(1);
    par_task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&par_output));
    par_task_data->outputs_count.emplace_back(1);
  }

  auto par_task = std::make_shared<ParallelTask>(par_task_data, func);

  auto perf_attributes = std::make_shared<ppc::core::PerfAttr>();
  perf_attributes->num_running = 10;
  auto start = boost::mpi::timer();
  perf_attributes->current_timer = [&] { return start.elapsed(); };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(par_task);

  switch (run_type) {
    case RunType::PIPELINE:
      perf_analyzer->pipeline_run(perf_attributes, perf_results);
      break;
    case RunType::TASK:
      perf_analyzer->task_run(perf_attributes, perf_results);
      break;
  }

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perf_results);
    EXPECT_NEAR(want, par_output, tolerance);
  }
}

}  // namespace chernykh_a_multidimensional_integral_simpson_mpi

namespace chernykh_a_mis_mpi = chernykh_a_multidimensional_integral_simpson_mpi;

TEST(chernykh_a_multidimensional_integral_simpson_mpi, test_pipeline_run) {
  auto func = [](const chernykh_a_mis_mpi::func_args_t &args) -> double {
    return std::exp(-args[0] - args[1] - args[2]) * std::sin(args[0]) * std::sin(args[1]) * std::sin(args[2]);
  };
  chernykh_a_mis_mpi::bounds_t bounds = {{0.0, std::numbers::pi}, {0.0, std::numbers::pi}, {0.0, std::numbers::pi}};
  chernykh_a_mis_mpi::step_range_t step_range = {2, 1000};
  double tolerance = 1e-6;
  double want = std::pow((1.0 + std::exp(-std::numbers::pi)) / 2.0, 3);
  chernykh_a_mis_mpi::run_task(chernykh_a_mis_mpi::RunType::PIPELINE, func, bounds, step_range, tolerance, want);
}

TEST(chernykh_a_multidimensional_integral_simpson_mpi, test_task_run) {
  auto func = [](const chernykh_a_mis_mpi::func_args_t &args) -> double {
    return std::exp(-args[0] - args[1] - args[2]) * std::sin(args[0]) * std::sin(args[1]) * std::sin(args[2]);
  };
  chernykh_a_mis_mpi::bounds_t bounds = {{0.0, std::numbers::pi}, {0.0, std::numbers::pi}, {0.0, std::numbers::pi}};
  chernykh_a_mis_mpi::step_range_t step_range = {2, 1000};
  double tolerance = 1e-6;
  double want = std::pow((1.0 + std::exp(-std::numbers::pi)) / 2.0, 3);
  chernykh_a_mis_mpi::run_task(chernykh_a_mis_mpi::RunType::TASK, func, bounds, step_range, tolerance, want);
}
