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
double octupleProductFunction(std::vector<double>& args) {
  return args[0] * args[1] * args[2] * args[3] * args[4] * args[5] * args[6] * args[7];
}
}  // namespace shvedova_v_mult_int_simpson_mpi

TEST(shvedova_v_mult_int_simpson_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  std::deque<std::pair<double, double>> integrationLimits = {{0.0, 10.0}, {0.0, 9.0}, {0.0, 8.0}, {0.0, 7.0},
                                                             {0.0, 6.0},  {0.0, 5.0}, {0.0, 4.0}, {0.0, 3.0}};
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

  std::function<double(std::vector<double>&)> func = shvedova_v_mult_int_simpson_mpi::octupleProductFunction;

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
    double result = *reinterpret_cast<double*>(taskDataPar->outputs[0]);
    EXPECT_TRUE(result > 0.0);
  }
}

TEST(shvedova_v_mult_int_simpson_mpi, test_task_run) {
  boost::mpi::communicator world;

  std::deque<std::pair<double, double>> integrationLimits = {{0.0, 10.0}, {0.0, 9.0}, {0.0, 8.0}, {0.0, 7.0},
                                                             {0.0, 6.0},  {0.0, 5.0}, {0.0, 4.0}, {0.0, 3.0}};
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

  std::function<double(std::vector<double>&)> func = shvedova_v_mult_int_simpson_mpi::octupleProductFunction;

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
    double result = *reinterpret_cast<double*>(taskDataPar->outputs[0]);
    EXPECT_TRUE(result > 0.0);
  }
}