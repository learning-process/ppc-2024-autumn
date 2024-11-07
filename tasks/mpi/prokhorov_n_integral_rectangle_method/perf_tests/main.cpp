#define _USE_MATH_DEFINES
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <cmath>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/prokhorov_n_integral_rectangle_method/include/ops_mpi.hpp"

TEST(prokhorov_n_integral_rectangle_method_mpi, test_pipeline_run_integration) {
  boost::mpi::communicator world;
  double left = 0.0, right = 1.0;
  int n = 1000;
  double global_result = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<double> inputs{left, right, static_cast<double>(n)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputs.data()));
    taskDataPar->inputs_count.emplace_back(inputs.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_result));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testMpiTaskParallel =
      std::make_shared<prokhorov_n_integral_rectangle_method_mpi::TestMPITaskParallel>(taskDataPar);
  testMpiTaskParallel->set_function([](double x) { return std::sin(x); });

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
    ASSERT_NEAR(global_result, 1 - std::cos(1.0), 0.01);
  }
}

TEST(prokhorov_n_integral_rectangle_method_mpi, test_task_run_integration) {
  boost::mpi::communicator world;
  double left = 0.0, right = M_PI;
  int n = 1000;
  double global_result = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<double> inputs{left, right, static_cast<double>(n)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputs.data()));
    taskDataPar->inputs_count.emplace_back(inputs.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_result));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testMpiTaskParallel =
      std::make_shared<prokhorov_n_integral_rectangle_method_mpi::TestMPITaskParallel>(taskDataPar);
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
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_NEAR(global_result, 0.0, 0.01);
  }
}
TEST(prokhorov_n_integral_rectangle_method_mpi, test_pipeline_run_exp) {
  boost::mpi::communicator world;
  double left = 0.0, right = 1.0;
  int n = 2000;
  double global_result = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<double> inputs{left, right, static_cast<double>(n)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputs.data()));
    taskDataPar->inputs_count.emplace_back(inputs.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_result));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testMpiTaskParallel =
      std::make_shared<prokhorov_n_integral_rectangle_method_mpi::TestMPITaskParallel>(taskDataPar);
  testMpiTaskParallel->set_function([](double x) { return std::exp(x); });

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
    ASSERT_NEAR(global_result, std::exp(1.0) - 1, 0.01);
  }
}

TEST(prokhorov_n_integral_rectangle_method_mpi, test_task_run_arctan) {
  boost::mpi::communicator world;
  double left = 0.0, right = 1.0;
  int n = 1500;
  double global_result = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<double> inputs{left, right, static_cast<double>(n)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputs.data()));
    taskDataPar->inputs_count.emplace_back(inputs.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_result));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testMpiTaskParallel =
      std::make_shared<prokhorov_n_integral_rectangle_method_mpi::TestMPITaskParallel>(taskDataPar);
  testMpiTaskParallel->set_function([](double x) { return 1.0 / (1.0 + x * x); });

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
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_NEAR(global_result, std::atan(1.0), 0.01);
  }
}

TEST(prokhorov_n_integral_rectangle_method_mpi, test_pipeline_run_sqrt) {
  boost::mpi::communicator world;
  double left = 0.0, right = 1.0;
  int n = 2500;
  double global_result = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<double> inputs{left, right, static_cast<double>(n)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputs.data()));
    taskDataPar->inputs_count.emplace_back(inputs.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_result));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testMpiTaskParallel =
      std::make_shared<prokhorov_n_integral_rectangle_method_mpi::TestMPITaskParallel>(taskDataPar);
  testMpiTaskParallel->set_function([](double x) { return std::sqrt(x); });

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
    ASSERT_NEAR(global_result, 2.0 / 3.0, 0.01);
  }
}
