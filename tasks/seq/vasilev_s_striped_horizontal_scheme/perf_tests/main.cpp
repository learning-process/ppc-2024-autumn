#include <gtest/gtest.h>

#include <chrono>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/vasilev_s_striped_horizontal_scheme/include/ops_seq.hpp"

TEST(vasilev_s_striped_horizontal_scheme_seq, Performance_Pipeline_Run) {
  int num_rows = 1000;
  int num_cols = 1000;
  std::vector<int> input_matrix(num_rows * num_cols);
  std::vector<int> input_vector(num_cols);
  std::vector<int> output_result(num_rows, 0);

  for (int i = 0; i < num_rows * num_cols; ++i) {
    input_matrix[i] = i % 100;
  }
  for (int i = 0; i < num_cols; ++i) {
    input_vector[i] = i % 50;
  }

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  taskDataSeq->inputs_count.emplace_back(input_vector.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  auto taskSequential =
      std::make_shared<vasilev_s_striped_horizontal_scheme_seq::StripedHorizontalSchemeSequential>(taskDataSeq);
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

  std::vector<int> expected_result(num_rows, 0);
  for (int i = 0; i < num_rows; ++i) {
    int sum = 0;
    for (int j = 0; j < num_cols; ++j) {
      sum += input_matrix[i * num_cols + j] * input_vector[j];
    }
    expected_result[i] = sum;
  }
  ASSERT_EQ(output_result, expected_result);
}

TEST(vasilev_s_striped_horizontal_scheme_seq, Performance_Task_Run) {
  int num_rows = 1000;
  int num_cols = 1000;
  std::vector<int> input_matrix(num_rows * num_cols);
  std::vector<int> input_vector(num_cols);
  std::vector<int> output_result(num_rows, 0);

  for (int i = 0; i < num_rows * num_cols; ++i) {
    input_matrix[i] = i % 100;
  }
  for (int i = 0; i < num_cols; ++i) {
    input_vector[i] = i % 50;
  }

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  taskDataSeq->inputs_count.emplace_back(input_vector.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_result.data()));
  taskDataSeq->outputs_count.emplace_back(output_result.size());

  auto taskSequential =
      std::make_shared<vasilev_s_striped_horizontal_scheme_seq::StripedHorizontalSchemeSequential>(taskDataSeq);
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

  std::vector<int> expected_result(num_rows, 0);
  for (int i = 0; i < num_rows; ++i) {
    int sum = 0;
    for (int j = 0; j < num_cols; ++j) {
      sum += input_matrix[i * num_cols + j] * input_vector[j];
    }
    expected_result[i] = sum;
  }
  ASSERT_EQ(output_result, expected_result);
}
