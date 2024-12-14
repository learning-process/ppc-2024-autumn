#include <gtest/gtest.h>

#include <chrono>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/vasilev_s_gaus3x3/include/ops_seq.hpp"

TEST(vasilev_s_gaus3x3_seq, pipeline_run) {
  int num_rows = 500;
  int num_cols = 500;
  std::vector<double> input_matrix(num_rows * num_cols, 1.0);
  std::vector<double> output_result(num_rows * num_cols);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_rows));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_cols));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  auto taskSequential = std::make_shared<vasilev_s_gaus3x3_seq::Gaus3x3Sequential>(taskDataSeq);
  ASSERT_TRUE(taskSequential->validation());
  taskSequential->pre_processing();
  taskSequential->run();
  taskSequential->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  auto start_time = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&start_time] {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start_time;
    return elapsed.count();
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);

  std::vector<double> expected_result(num_rows * num_cols, 1.0);

  // Верхняя и нижняя строки.
  for (int i = 0; i < num_cols; i++) {
    expected_result[i] = 0.0;                              // Верхняя строка.
    expected_result[(num_rows - 1) * num_cols + i] = 0.0;  // Нижняя строка.
  }

  // Левые и правые границы.
  for (int i = 1; i < num_rows - 1; i++) {
    expected_result[i * num_cols] = 0.0;                   // Левая граница.
    expected_result[i * num_cols + (num_cols - 1)] = 0.0;  // Правая граница.
  }
  EXPECT_EQ(output_result, expected_result);
}

TEST(vasilev_s_gaus3x3_seq, task_run) {
  int num_rows = 500;
  int num_cols = 500;
  std::vector<double> input_matrix(num_rows * num_cols, 1.0);
  std::vector<double> output_result(num_rows * num_cols);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_rows));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_cols));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  auto taskSequential = std::make_shared<vasilev_s_gaus3x3_seq::Gaus3x3Sequential>(taskDataSeq);
  ASSERT_TRUE(taskSequential->validation());
  taskSequential->pre_processing();
  taskSequential->run();
  taskSequential->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  auto start_time = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&start_time] {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start_time;
    return elapsed.count();
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);

  std::vector<double> expected_result(num_rows * num_cols, 1.0);

  // Верхняя и нижняя строки.
  for (int i = 0; i < num_cols; i++) {
    expected_result[i] = 0.0;                              // Верхняя строка.
    expected_result[(num_rows - 1) * num_cols + i] = 0.0;  // Нижняя строка.
  }

  // Левые и правые границы.
  for (int i = 1; i < num_rows - 1; i++) {
    expected_result[i * num_cols] = 0.0;                   // Левая граница.
    expected_result[i * num_cols + (num_cols - 1)] = 0.0;  // Правая граница.
  }
  EXPECT_EQ(output_result, expected_result);
}
