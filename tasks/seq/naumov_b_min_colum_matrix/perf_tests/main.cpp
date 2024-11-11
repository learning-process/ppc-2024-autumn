#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/naumov_b_min_colum_matrix/include/ops_seq.hpp"

TEST(naumov_b_min_colum_matrix_seq, test_pipeline_run) {
  const int rows = 1000;
  const int cols = 1000;

  std::vector<int> in(rows * cols);
  std::generate(in.begin(), in.end(), []() { return rand() % 100; });

  std::vector<int> out(cols, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count = {rows, cols};
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count = {cols};

  auto testTaskSequential = std::make_shared<naumov_b_min_colum_matrix_seq::TestTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;

  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(out.size(), static_cast<size_t>(cols));
}

TEST(snaumov_b_min_colum_matrix_seq, test_task_run) {
  const int rows = 1000;
  const int cols = 1000;

  std::vector<int> in(rows * cols);
  std::generate(in.begin(), in.end(), []() { return rand() % 100; });

  std::vector<int> out(cols, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count = {rows, cols};
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count = {cols};

  auto testTaskSequential = std::make_shared<naumov_b_min_colum_matrix_seq::TestTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;

  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;  // Convert to seconds
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(out.size(), static_cast<size_t>(cols));
}
