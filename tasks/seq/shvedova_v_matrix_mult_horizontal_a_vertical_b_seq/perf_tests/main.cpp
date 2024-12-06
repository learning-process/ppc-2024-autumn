#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/shvedova_v_matrix_mult_horizontal_a_vertical_b_seq/include/ops_seq.hpp"

TEST(shvedova_v_matrix_mult_horizontal_a_vertical_b_seq, test_pipeline_run) {
  const int row_a = 1000;
  const int col_a = 1000;
  const int row_b = 1000;
  const int col_b = 1000;

  std::vector<int> matrix_a(row_a * col_a, 1);
  std::vector<int> matrix_b(row_b * col_b, 1);
  std::vector<int> matrix_c(row_a * col_b, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  size_t sizes[] = {row_a, col_a, row_b, col_b};

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(sizes));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_c.data()));
  taskDataSeq->inputs_count = {4, row_a * col_a, row_b * col_b};
  taskDataSeq->outputs_count = {row_a * col_b};

  auto matrixMultiplicationTaskSequential =
      std::make_shared<shvedova_v_matrix_mult_horizontal_a_vertical_b_seq::MatrixMultiplicationTaskSequential>(
          taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(matrixMultiplicationTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  size_t answer = static_cast<size_t>(row_a) * static_cast<size_t>(col_b);

  ASSERT_EQ(answer, matrix_c.size());
}

TEST(shvedova_v_matrix_mult_horizontal_a_vertical_b_seq, test_task_run) {
  const int row_a = 1000;
  const int col_a = 1000;
  const int row_b = 1000;
  const int col_b = 1000;

  std::vector<int> matrix_a(row_a * col_a, 1);
  std::vector<int> matrix_b(row_b * col_b, 1);
  std::vector<int> matrix_c(row_a * col_b, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  size_t sizes[] = {row_a, col_a, row_b, col_b};

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(sizes));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_c.data()));
  taskDataSeq->inputs_count = {4, row_a * col_a, row_b * col_b};
  taskDataSeq->outputs_count = {row_a * col_b};

  auto matrixMultiplicationTaskSequential =
      std::make_shared<shvedova_v_matrix_mult_horizontal_a_vertical_b_seq::MatrixMultiplicationTaskSequential>(
          taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(matrixMultiplicationTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  size_t answer = static_cast<size_t>(row_a) * static_cast<size_t>(col_b);

  ASSERT_EQ(answer, matrix_c.size());
}