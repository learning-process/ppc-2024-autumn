#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <cmath>
#include <deque>
#include <functional>
#include <memory>

#include "core/perf/include/perf.hpp"
#include "mpi/shvedova_v_mult_int_simpson/include/ops_mpi.hpp"

namespace shvedova_v_mult_int_simpson_mpi {
double tripleComplexFunction(std::vector<double>& args) {
  double sum_of_squares = 0.0;
  double sum_of_linear = 0.0;
  for (size_t i = 0; i < args.size(); ++i) {
    sum_of_squares += args[i] * args[i];
    sum_of_linear += args[i];
  }
  return std::sin(sum_of_squares) * std::exp(-sum_of_linear);
}
}  // namespace shvedova_v_mult_int_simpson_mpi

TEST(shvedova_v_mult_int_simpson_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  std::deque<std::pair<double, double>> integrationLimits = {{0.0, 10.0}, {0.0, 9.0}, {0.0, 8.0}};
  double precision = 0.001;

  std::vector<std::pair<double, double>> limitsVec(integrationLimits.begin(), integrationLimits.end());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double out = 0.0;
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(limitsVec.data()));
    taskDataPar->inputs_count.emplace_back(limitsVec.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&precision));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&out));
    taskDataPar->outputs_count.emplace_back(1);
  }

  std::function<double(std::vector<double>&)> func = shvedova_v_mult_int_simpson_mpi::tripleComplexFunction;

  auto parallelTask = std::make_shared<shvedova_v_mult_int_simpson_mpi::SimpsonMultIntParallel>(taskDataPar, func);

  ASSERT_TRUE(parallelTask->validation());
  parallelTask->pre_processing();
  parallelTask->run();
  parallelTask->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  boost::mpi::timer timer;
  perfAttr->current_timer = [&] { return timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(parallelTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    EXPECT_TRUE(out > 0.0);
  }
}

TEST(shvedova_v_mult_int_simpson_mpi, test_task_run) {
  boost::mpi::communicator world;

  std::deque<std::pair<double, double>> integrationLimits = {{0.0, 10.0}, {0.0, 9.0}, {0.0, 8.0}};
  double precision = 0.001;

  std::vector<std::pair<double, double>> limitsVec(integrationLimits.begin(), integrationLimits.end());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double out = 0.0;
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(limitsVec.data()));
    taskDataPar->inputs_count.emplace_back(limitsVec.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&precision));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&out));
    taskDataPar->outputs_count.emplace_back(1);
  }

  std::function<double(std::vector<double>&)> func = shvedova_v_mult_int_simpson_mpi::tripleComplexFunction;

  auto parallelTask = std::make_shared<shvedova_v_mult_int_simpson_mpi::SimpsonMultIntParallel>(taskDataPar, func);

  ASSERT_TRUE(parallelTask->validation());
  parallelTask->pre_processing();
  parallelTask->run();
  parallelTask->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  boost::mpi::timer timer;
  perfAttr->current_timer = [&] { return timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(parallelTask);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    EXPECT_TRUE(out > 0.0);
  }
}
