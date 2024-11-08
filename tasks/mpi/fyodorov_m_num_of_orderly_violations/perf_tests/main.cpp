// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/timer.hpp>
#include <chrono>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/fyodorov_m_num_of_orderly_violations/include/ops_mpi.hpp"

TEST(fyodorov_m_num_of_orderly_violations_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int count_size_vector = 2000000;
  std::vector<int> global_vec(count_size_vector, 0);
  std::vector<int32_t> global_violations(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (int i = 0; i < count_size_vector; i++) {
      global_vec[i] = i;
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_violations.data()));
    taskDataPar->outputs_count.emplace_back(global_violations.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(0, global_violations[0]);
  }
}

TEST(fyodorov_m_num_of_orderly_violations_mpi, test_task_run) {
  boost::mpi::communicator world;
  int count_size_vector = 2000000;
  std::vector<int> global_vec(count_size_vector, 0);
  std::vector<int32_t> global_violations(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (int i = 0; i < count_size_vector; i++) {
      global_vec[i] = i;
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_violations.data()));
    taskDataPar->outputs_count.emplace_back(global_violations.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(0, global_violations[0]);
  }
}
