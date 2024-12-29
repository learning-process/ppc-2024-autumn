// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/vladimirova_j_jarvis_method/include/ops_seq.hpp"

namespace mpi_vladimirova_j_jarvis_method_seq {
std::vector<int> getRandomVal(size_t col, size_t row, size_t n) {
  std::vector<int> ans(row * col, 255);

  std::random_device dev;
  std::mt19937 gen(dev());
  for (size_t i = 0; i < n; i++) {
    int c = gen() % col;
    int r = gen() % row;
    ans[r * row + c] = 0;
  }
  return ans;
}

}  // namespace mpi_vladimirova_j_jarvis_method_seq

TEST(sequential_vladimirova_j_jarvis_method_perf_test, test_pipeline_run) {
  const int count = 6000;
  const int sz = 15000;
  // Create data
  std::vector<int> in = mpi_vladimirova_j_jarvis_method_seq::getRandomVal(sz, sz, count / 2);
  std::vector<int> out(count, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(sz);
  taskDataSeq->inputs_count.emplace_back(sz);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<vladimirova_j_jarvis_method_seq::TestTaskSequential>(taskDataSeq);

  ASSERT_EQ(testTaskSequential->validation(), true);

  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();
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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(taskDataSeq->outputs_count[0] <= count, true);
}

TEST(sequential_vladimirova_j_jarvis_method_perf_test, test_task_run) {
  const int count = 6000;
  const int sz = 15000;
  // Create data
  std::vector<int> in = mpi_vladimirova_j_jarvis_method_seq::getRandomVal(sz, sz, count / 2);
  std::vector<int> out(count, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(sz);
  taskDataSeq->inputs_count.emplace_back(sz);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<vladimirova_j_jarvis_method_seq::TestTaskSequential>(taskDataSeq);
  ASSERT_EQ(testTaskSequential->validation(), true);

  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(taskDataSeq->outputs_count[0] <= count, true);
}