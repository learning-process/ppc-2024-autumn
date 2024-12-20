// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/varfolomeev_g_transfer_from_one_to_all_scatter_custom/include/ops_mpi.hpp"

static std::vector<int> getRandomVector_Custom(int sz, int a, int b) {  // [a, b]
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % (b - a + 1) + a;
  }
  return vec;
}

TEST(mpi_varfolomeev_g_transfer_from_one_to_all_custom_scatter_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_max(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    int count_size_vector = 10000000;
    global_vec = getRandomVector_Custom(count_size_vector, 0, 5);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  auto MyScatterTestMPITaskParallel =
      std::make_shared<varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi::MyScatterTestMPITaskParallel>(
          taskDataPar, "max");
  ASSERT_EQ(MyScatterTestMPITaskParallel->validation(), true);
  MyScatterTestMPITaskParallel->pre_processing();
  MyScatterTestMPITaskParallel->run();
  MyScatterTestMPITaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MyScatterTestMPITaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(5, global_max[0]);
  }
}

TEST(mpi_varfolomeev_g_transfer_from_one_to_all_custom_scatter_perf_test, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_max(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 10000000;
    global_vec = getRandomVector_Custom(count_size_vector, 0, 5);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  auto MyScatterTestMPITaskParallel =
      std::make_shared<varfolomeev_g_transfer_from_one_to_all_custom_scatter_mpi::MyScatterTestMPITaskParallel>(
          taskDataPar, "max");
  ASSERT_EQ(MyScatterTestMPITaskParallel->validation(), true);
  MyScatterTestMPITaskParallel->pre_processing();
  MyScatterTestMPITaskParallel->run();
  MyScatterTestMPITaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MyScatterTestMPITaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(5, global_max[0]);
  }
}
