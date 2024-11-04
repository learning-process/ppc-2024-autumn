#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/gordeva_t_max_val_of_column_matrix/include/ops_seq.hpp"

TEST(gordeva_t_max_val_of_column_matrix_seq, test_pipeline_run) {
  const int cols = 5000;
  const int rows = 5000;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  auto testTaskSequential = std::make_shared<gordeva_t_max_val_of_column_matrix_seq::TestTaskSequential>(taskDataSeq);

  std::vector<std::vector<int>> matrix =
      gordeva_t_max_val_of_column_matrix_seq::TestTaskSequential::rand_matr(rows, cols);

  for (auto &i : matrix) taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(i.data()));

  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  std::vector<int> res_vec(cols, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_vec.data()));
  taskDataSeq->outputs_count.emplace_back(res_vec.size());

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

  for (int i = 0; i < cols; i++) ASSERT_EQ(res_vec[i], 200);
}

TEST(gordeva_t_max_val_of_column_matrix_seq, test_task_run) {
  const int cols = 7000;
  const int rows = 7000;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  auto testTaskSequential = std::make_shared<gordeva_t_max_val_of_column_matrix_seq::TestTaskSequential>(taskDataSeq);

  std::vector<std::vector<int>> matr_rand =
      gordeva_t_max_val_of_column_matrix_seq::TestTaskSequential::rand_matr(rows, cols);
  for (auto &row : matr_rand) taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));

  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);

  std::vector<int> res_vec(cols, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_vec.data()));
  taskDataSeq->outputs_count.emplace_back(res_vec.size());

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
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  for (int i = 0; i < cols; i++) ASSERT_EQ(res_vec[i], 200);
}
