// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/malyshev_a_sum_rows_matrix/include/ops_seq.hpp"

namespace malyshev_a_sum_rows_matrix_test_function {

std::vector<std::vector<int32_t>> getRandomData(uint32_t rows, uint32_t cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<std::vector<int32_t>> data(rows, std::vector<int32_t>(cols));

  for (auto &row : data) {
    for (auto &el : row) {
      el = -200 + gen() % (300 + 200 + 1);
    }
  }

  return data;
}

}  // namespace malyshev_a_sum_rows_matrix_test_function

TEST(malyshev_a_sum_rows_matrix_seq, test_pipeline_run) {
  uint32_t rows = 3000;
  uint32_t cols = 3000;

  std::vector<std::vector<int32_t>> randomData;
  std::vector<int32_t> seqSum;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  randomData = malyshev_a_sum_rows_matrix_test_function::getRandomData(rows, cols);
  seqSum.resize(rows);

  for (auto &row : randomData) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
  }

  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->inputs_count.push_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqSum.data()));
  taskDataSeq->outputs_count.push_back(rows);

  auto taskSeq = std::make_shared<malyshev_a_sum_rows_matrix_seq::TestTaskSequential>(taskDataSeq);

  ASSERT_TRUE(taskSeq->validation());
  ASSERT_TRUE(taskSeq->pre_processing());
  ASSERT_TRUE(taskSeq->run());
  ASSERT_TRUE(taskSeq->post_processing());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskSeq);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(malyshev_a_sum_rows_matrix_seq, test_task_run) {
  uint32_t rows = 3000;
  uint32_t cols = 3000;

  std::vector<std::vector<int32_t>> randomData;
  std::vector<int32_t> seqSum;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  randomData = malyshev_a_sum_rows_matrix_test_function::getRandomData(rows, cols);
  seqSum.resize(rows);

  for (auto &row : randomData) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
  }

  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->inputs_count.push_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqSum.data()));
  taskDataSeq->outputs_count.push_back(rows);

  auto taskSeq = std::make_shared<malyshev_a_sum_rows_matrix_seq::TestTaskSequential>(taskDataSeq);

  ASSERT_TRUE(taskSeq->validation());
  ASSERT_TRUE(taskSeq->pre_processing());
  ASSERT_TRUE(taskSeq->run());
  ASSERT_TRUE(taskSeq->post_processing());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskSeq);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}