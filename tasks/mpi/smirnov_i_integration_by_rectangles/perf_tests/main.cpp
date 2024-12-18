#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/smirnov_i_integration_by_rectangles/include/ops_mpi.hpp"
double f1(double x) { return x * x; }
TEST(smirnov_i_integration_by_rectangles_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  double left = 0.0;
  double right = 1.0;
  int n_ = 1000;
  std::vector<double> global_res(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&left));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&right));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n_));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->inputs_count.emplace_back(3);
  }

  auto testMpiTaskParallel = std::make_shared<smirnov_i_integration_by_rectangles::TestMPITaskParallel>(taskDataPar);
  testMpiTaskParallel->set_function(f1);

  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    double expected_result = 1. / 3;
    ASSERT_NEAR(global_res[0], expected_result, 1e-5);
  }
}
TEST(smirnov_i_integration_by_rectangles_mpi, test_task_run) {
  boost::mpi::communicator world;
  double left = 0.0;
  double right = 1.0;
  int n_ = 1000;
  std::vector<double> global_res(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&left));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&right));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n_));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->inputs_count.emplace_back(3);
  }

  auto testMpiTaskParallel = std::make_shared<smirnov_i_integration_by_rectangles::TestMPITaskParallel>(taskDataPar);
  testMpiTaskParallel->set_function(f1);

  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    double expected_result = 1. / 3;
    ASSERT_NEAR(global_res[0], expected_result, 1e-5);
  }
}
