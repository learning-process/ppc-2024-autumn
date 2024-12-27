// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/zinoviev_a_bellman_ford/include/ops_seq.hpp"

TEST(zinoviev_a_bellman_ford, test_pipeline_run) {
  std::vector<int> graph = {0, 4, 0, 0, 0, 0,  0, 8, 0, 4, 0,  8, 0, 0, 0,  0, 11, 0, 0, 8, 0, 7,  0,  4, 0, 0, 2,
                            0, 0, 7, 0, 9, 14, 0, 0, 0, 0, 0,  0, 9, 0, 10, 0, 0,  0, 0, 0, 4, 14, 10, 0, 2, 0, 0,
                            0, 0, 0, 0, 0, 2,  0, 1, 6, 8, 11, 0, 0, 0, 0,  1, 0,  7, 0, 0, 2, 0,  0,  0, 6, 7, 0};
  std::vector<int> shortest_paths(9, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  taskData->inputs_count.emplace_back(9);
  taskData->inputs_count.emplace_back(28);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(shortest_paths.data()));
  taskData->outputs_count.emplace_back(9);

  auto task = std::make_shared<zinoviev_a_bellman_ford_seq::BellmanFordSeq>(taskData);
  ASSERT_EQ(task->validation(), true);
  task->pre_processing();
  task->run();
  task->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(zinoviev_a_bellman_ford, test_task_run) {
  std::vector<int> graph = {0, 4, 0, 0, 0, 0,  0, 8, 0, 4, 0,  8, 0, 0, 0,  0, 11, 0, 0, 8, 0, 7,  0,  4, 0, 0, 2,
                            0, 0, 7, 0, 9, 14, 0, 0, 0, 0, 0,  0, 9, 0, 10, 0, 0,  0, 0, 0, 4, 14, 10, 0, 2, 0, 0,
                            0, 0, 0, 0, 0, 2,  0, 1, 6, 8, 11, 0, 0, 0, 0,  1, 0,  7, 0, 0, 2, 0,  0,  0, 6, 7, 0};
  std::vector<int> shortest_paths(9, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  taskData->inputs_count.emplace_back(9);
  taskData->inputs_count.emplace_back(28);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(shortest_paths.data()));
  taskData->outputs_count.emplace_back(9);

  auto task = std::make_shared<zinoviev_a_bellman_ford_seq::BellmanFordSeq>(taskData);
  ASSERT_EQ(task->validation(), true);
  task->pre_processing();
  task->run();
  task->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}