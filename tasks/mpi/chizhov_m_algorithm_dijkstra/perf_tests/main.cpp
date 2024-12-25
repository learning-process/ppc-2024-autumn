// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/chizhov_m_algorithm_dijkstra/include/ops_mpi.hpp"

TEST(chizhov_m_dijkstra_mpi_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;
  int count_size_vector = 1000;
  int st = 0;
  std::vector<int> global_matrix(count_size_vector * count_size_vector, 3);
  std::vector<int32_t> global_path(count_size_vector, 3);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    for (int i = 0; i < count_size_vector; i++) {
      global_matrix[i * count_size_vector + i] = 0;
    }
    global_path[0] = 0;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs_count.emplace_back(count_size_vector);
    taskDataPar->inputs_count.emplace_back(st);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_path.data()));
    taskDataPar->outputs_count.emplace_back(global_path.size());
  }

  auto testMpiTaskParallel = std::make_shared<chizhov_m_dijkstra_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
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
    ASSERT_EQ(3, global_path[3]);
  }
}

TEST(chizhov_m_dijkstra_mpi_perf_test, test_task_run) {
  boost::mpi::communicator world;
  int count_size_vector = 1000;
  int st = 5;
  std::vector<int> global_matrix(count_size_vector * count_size_vector, 3);
  std::vector<int32_t> global_path(count_size_vector, 3);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    for (int i = 0; i < count_size_vector; i++) {
      global_matrix[i * count_size_vector + i] = 0;
    }
    global_path[0] = 0;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs_count.emplace_back(count_size_vector);
    taskDataPar->inputs_count.emplace_back(st);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_path.data()));
    taskDataPar->outputs_count.emplace_back(global_path.size());
  }

  auto testMpiTaskParallel = std::make_shared<chizhov_m_dijkstra_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
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
    ASSERT_EQ(3, global_path[3]);
  }
}
