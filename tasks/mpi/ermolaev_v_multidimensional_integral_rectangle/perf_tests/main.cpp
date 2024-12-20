// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/ermolaev_v_multidimensional_integral_rectangle/include/ops_mpi.hpp"

namespace ermolaev_v_multidimensional_integral_rectangle_mpi {

void testBody(std::vector<std::pair<double, double>> limits,
              ermolaev_v_multidimensional_integral_rectangle_mpi::function func,
              ppc::core::PerfResults::TypeOfRunning type, double eps = 1e-4) {
  boost::mpi::communicator world;
  double out;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
    taskDataPar->inputs_count.emplace_back(limits.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
    taskDataPar->outputs_count.emplace_back(1);
  }

  // Create Task
  auto testTaskParallel =
      std::make_shared<ermolaev_v_multidimensional_integral_rectangle_mpi::TestMPITaskParallel>(taskDataPar, func);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  if (type == ppc::core::PerfResults::PIPELINE)
    perfAnalyzer->pipeline_run(perfAttr, perfResults);
  else if (type == ppc::core::PerfResults::TASK_RUN)
    perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

double testFunc(std::vector<double> &args) { return args.at(0); }
}  // namespace ermolaev_v_multidimensional_integral_rectangle_mpi
namespace erm_integral_mpi = ermolaev_v_multidimensional_integral_rectangle_mpi;

TEST(ermolaev_v_multidimensional_integral_rectangle_mpi, test_pipeline_run) {
  std::vector<std::pair<double, double>> limits(10, {-7, 7});
  erm_integral_mpi::testBody(limits, erm_integral_mpi::testFunc, ppc::core::PerfResults::PIPELINE);
}

TEST(ermolaev_v_multidimensional_integral_rectangle_mpi, test_task_run) {
  std::vector<std::pair<double, double>> limits(10, {-7, 7});
  erm_integral_mpi::testBody(limits, erm_integral_mpi::testFunc, ppc::core::PerfResults::TASK_RUN);
}
