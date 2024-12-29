#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm/include/ops_seq.hpp"

TEST(alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm_seq, test_pipeline_run) {
  std::vector<double> A(1000 * 1000, 0);
  int row_A = 1000;
  int column_A = 1000;
  std::vector<double> B(1000 * 1000, 0);
  int row_B = 1000;
  int column_B = 1000;
  std::vector<double> out(1000 * 1000);

  std::vector<double> ans(1000 * 1000, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(row_A);
  taskDataSeq->inputs_count.emplace_back(column_A);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(row_B);
  taskDataSeq->inputs_count.emplace_back(column_B);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
                           dense_matrix_multiplication_block_scheme_fox_algorithm_seq>(taskDataSeq);

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

  ASSERT_EQ(ans, out);
}

TEST(alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm_seq, test_task_run) {
  std::vector<double> A(1000 * 1000, 0);
  int row_A = 1000;
  int column_A = 1000;
  std::vector<double> B(1000 * 1000, 0);
  int row_B = 1000;
  int column_B = 1000;
  std::vector<double> out(1000 * 1000);

  std::vector<double> ans(1000 * 1000, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(row_A);
  taskDataSeq->inputs_count.emplace_back(column_A);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(row_B);
  taskDataSeq->inputs_count.emplace_back(column_B);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
                           dense_matrix_multiplication_block_scheme_fox_algorithm_seq>(taskDataSeq);

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

  ASSERT_EQ(ans, out);
}