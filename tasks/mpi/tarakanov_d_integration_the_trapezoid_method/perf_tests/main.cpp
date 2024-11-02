// Copyright 2024 Tarakanov Denis
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/tarakanov_d_integration_the_trapezoid_method/include/ops_mpi.hpp"

TEST(tarakanov_d_integration_the_trapezoid_method_mpi_perf_tests, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<double> global_res(1, 0.0);

  double a = 0.0;
  double b = 1.0;
  double h = 1e-8;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&h));
    taskDataPar->inputs_count.emplace_back(3);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto parallelTask =
      std::make_shared<tarakanov_d_integration_the_trapezoid_method_mpi::integration_the_trapezoid_method_par>(
          taskDataPar);

  ASSERT_EQ(parallelTask->validation(), true);
  parallelTask->pre_processing();
  parallelTask->run();
  parallelTask->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(parallelTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    double expected_value = 0.335;
    ASSERT_NEAR(expected_value, global_res[0], 0.1);
  }
}

TEST(tarakanov_d_integration_the_trapezoid_method_mpi_perf_tests, test_task_run) {
  boost::mpi::communicator world;
  std::vector<double> global_res(1, 0.0);

  double a = 0.0;
  double b = 1.0;
  double h = 1e-8;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&h));
    taskDataPar->inputs_count.emplace_back(3);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto parallelTask =
      std::make_shared<tarakanov_d_integration_the_trapezoid_method_mpi::integration_the_trapezoid_method_par>(
          taskDataPar);

  ASSERT_EQ(parallelTask->validation(), true);
  parallelTask->pre_processing();
  parallelTask->run();
  parallelTask->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(parallelTask);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    double expected_value = 0.335;
    ASSERT_NEAR(expected_value, global_res[0], 0.1);
  }
}