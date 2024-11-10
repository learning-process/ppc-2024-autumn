// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/makhov_m_num_of_diff_elements_in_two_str/include/ops_mpi.hpp"

TEST(mpi_makhov_m_num_of_diff_elements_in_two_str_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;
  std::string str1;
  std::string str2;
  std::vector<int32_t> global_sum(1, 0);
  std::random_device dev;
  std::mt19937 gen(dev());
  const size_t size = 10000000;
  char min = '0';
  char max = '9';

  // Create data
  for (size_t i = 0; i < size; i++) {
    str1 += (char)(min + gen() % (max - min + 1));
    str2 += (char)(min + gen() % (max - min + 1));
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str1.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str2.data()));
    taskDataPar->inputs_count.emplace_back(str1.size());
    taskDataPar->inputs_count.emplace_back(str2.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<makhov_m_num_of_diff_elements_in_two_str_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel->validation());
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(mpi_makhov_m_num_of_diff_elements_in_two_str_perf_test, test_task_run) {
  boost::mpi::communicator world;
  std::string str1;
  std::string str2;
  std::vector<int32_t> global_sum(1, 0);
  std::random_device dev;
  std::mt19937 gen(dev());
  const size_t size = 10000000;
  char min = '0';
  char max = '9';

  // Create data
  for (size_t i = 0; i < size; i++) {
    str1 += (char)(min + gen() % (max - min + 1));
    str2 += (char)(min + gen() % (max - min + 1));
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str1.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str2.data()));
    taskDataPar->inputs_count.emplace_back(str1.size());
    taskDataPar->inputs_count.emplace_back(str2.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<makhov_m_num_of_diff_elements_in_two_str_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel->validation());
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}
