#define _USE_MATH_DEFINES

#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <functional>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/prokhorov_n_rectangular_integration/include/ops_mpi.hpp"

TEST(prokhorov_n_rectangular_integration_mpi, test_pipeline_run_cos_x) {
  boost::mpi::communicator world;
  std::vector<double> global_input = {0.0, M_PI / 2.0, 100000.0};
  std::vector<double> global_result(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
  taskDataPar->inputs_count.emplace_back(global_input.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  taskDataPar->outputs_count.emplace_back(global_result.size());

  auto testMpiTaskParallel =
      std::make_shared<prokhorov_n_rectangular_integration_mpi::TestMPITaskParallel>(taskDataPar);
  testMpiTaskParallel->set_function([](double x) { return std::cos(x); });

  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_NEAR(global_result[0], 1.0, 1e-3);
  }
}

TEST(prokhorov_n_rectangular_integration_mpi, test_pipeline_run_x_cubed) {
  boost::mpi::communicator world;
  std::vector<double> global_input = {0.0, 1.0, 100000.0};
  std::vector<double> global_result(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
  taskDataPar->inputs_count.emplace_back(global_input.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  taskDataPar->outputs_count.emplace_back(global_result.size());

  auto testMpiTaskParallel =
      std::make_shared<prokhorov_n_rectangular_integration_mpi::TestMPITaskParallel>(taskDataPar);
  testMpiTaskParallel->set_function([](double x) { return x * x * x; });

  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_NEAR(global_result[0], 0.25, 1e-3);
  }
}

TEST(prokhorov_n_rectangular_integration_mpi, test_pipeline_run_exp_minus_x_squared) {
  boost::mpi::communicator world;
  std::vector<double> global_input = {0.0, 1.0, 100000.0};
  std::vector<double> global_result(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
  taskDataPar->inputs_count.emplace_back(global_input.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  taskDataPar->outputs_count.emplace_back(global_result.size());

  auto testMpiTaskParallel =
      std::make_shared<prokhorov_n_rectangular_integration_mpi::TestMPITaskParallel>(taskDataPar);
  testMpiTaskParallel->set_function([](double x) { return std::exp(-x * x); });

  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_NEAR(global_result[0], 0.746824, 1e-3);
  }
}
