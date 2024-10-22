// Copyright 2024 Lupsha Egor

#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/lupsha_e_rect_integration/include/ops_mpi.hpp"

TEST(lupsha_e_rect_integration_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  double lower_bound = 0.0;
  double upper_bound = 1.0;
  int num_intervals = 1000000;
  std::vector<double> global_sum(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_bound));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_bound));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_intervals));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  auto testMpiTaskParallel = std::make_shared<lupsha_e_rect_integration_mpi::TestMPITaskParallel>(taskDataPar);
  testMpiTaskParallel->function_set([](double x) { return x * x; });

  ASSERT_TRUE(testMpiTaskParallel->validation());
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
    double expected_result = (upper_bound * upper_bound * upper_bound - lower_bound * lower_bound * lower_bound) / 3;
    ASSERT_NEAR(global_sum[0], expected_result, 1e-5);
  }
}

TEST(lupsha_e_rect_integration_mpi, test_task_run) {
  boost::mpi::communicator world;
  double lower_bound = 0.0;
  double upper_bound = 1.0;
  int num_intervals = 1000000;
  std::vector<double> global_sum(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_bound));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_bound));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_intervals));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  auto testMpiTaskParallel = std::make_shared<lupsha_e_rect_integration_mpi::TestMPITaskParallel>(taskDataPar);
  testMpiTaskParallel->function_set([](double x) { return x * x; });

  ASSERT_TRUE(testMpiTaskParallel->validation());
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
    double expected_result = (upper_bound * upper_bound * upper_bound - lower_bound * lower_bound * lower_bound) /
                             3;
    ASSERT_NEAR(global_sum[0], expected_result, 1e-5);
  }
}
int main(int argc, char** argv) {
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (world.rank() != 0) {
    delete listeners.Release(listeners.default_result_printer());
  }
  return RUN_ALL_TESTS();
}