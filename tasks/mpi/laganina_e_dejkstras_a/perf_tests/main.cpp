#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "mpi/laganina_e_dejkstras_a/include/ops_mpi.hpp"

TEST(laganina_e_dejkstras_a_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int v_ = 1000;
  // Create data
  std::vector<int> graph(v_ * v_, 0);
  for (int i = 0; i < v_ - 1; i++) {
    for (int j = 1; j < v_; j++) {
      graph[i * v_ + j] = 1;
    }
  }
  for (int k = 0; k < v_ * v_; k += (v_ + 1)) {
    graph[k] = 0;
  }
  std::vector<int> expectResult(v_, 0);
  for (int i = 1; i < v_; i++) {
    expectResult[i] = 1;
  }
  std::vector<int> trueResult(v_, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(v_);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(trueResult.data()));
    taskDataPar->outputs_count.emplace_back(trueResult.size());
  }
  // Create Task
  auto testTaskParallel = std::make_shared<laganina_e_dejskras_a_mpi::TestMPITaskParallel>(taskDataPar);

  ASSERT_TRUE(testTaskParallel->validation());
  testTaskParallel->pre_processing();
  testTaskParallel->run();
  testTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  if (world.rank() == 0) {
    ASSERT_EQ(expectResult, trueResult);
  }
}

TEST(laganina_e_dejkstras_a_mpi, test_task_run) {
  boost::mpi::communicator world;
  int v_ = 1000;
  // Create data
  std::vector<int> graph(v_ * v_, 0);
  for (int i = 0; i < v_ - 1; i++) {
    for (int j = 1; j < v_; j++) {
      graph[i * v_ + j] = 1;
    }
  }
  for (int k = 0; k < v_ * v_; k += (v_ + 1)) {
    graph[k] = 0;
  }
  std::vector<int> expectResult(v_, 0);
  for (int i = 1; i < v_; i++) {
    expectResult[i] = 1;
  }
  std::vector<int> trueResult(v_, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(v_);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(trueResult.data()));
    taskDataPar->outputs_count.emplace_back(trueResult.size());
  }
  // Create Task
  auto testTaskParallel = std::make_shared<laganina_e_dejskras_a_mpi::TestMPITaskParallel>(taskDataPar);

  ASSERT_TRUE(testTaskParallel->validation());
  testTaskParallel->pre_processing();
  testTaskParallel->run();
  testTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  if (world.rank() == 0) {
    ASSERT_EQ(expectResult, trueResult);
  }
}