#include <gtest/gtest.h>

#include <chrono>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/sedova_o_vertical_ribbon_scheme/include/ops_seq.hpp"

TEST(sedova_o_vertical_ribbon_scheme_seq, test_pipeline_run) {
  int rows_ = 2000;
  int cols_ = 2000;
  std::vector<int> input_matrix_(rows_ * cols_);
  std::vector<int> input_vector_(cols_);
  std::vector<int> result_vector_(rows_, 0);

  for (int cols = 0; cols < cols_; ++cols) {
    for (int rows = 0; rows < rows_; ++rows) {
      input_matrix_[rows + cols * rows_] = (rows + cols * rows_) % 100;
    }
  }
  for (int i = 0; i < cols_; ++i) {
    input_vector_[i] = i % 50;
  }

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_.data()));
  taskDataSeq->inputs_count.emplace_back(input_matrix_.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector_.data()));
  taskDataSeq->inputs_count.emplace_back(input_vector_.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_vector_.data()));
  taskDataSeq->outputs_count.emplace_back(result_vector_.size());

  auto taskSequential = std::make_shared<sedova_o_vertical_ribbon_scheme_seq::Sequential>(taskDataSeq);
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

  std::vector<int> expected_result(rows_, 0);
  for (int i = 0; i < rows_; ++i) {
    int sum = 0;
    for (int j = 0; j < cols_; ++j) {
      sum += input_matrix_[i + j * rows_] * input_vector_[j];
    }
    expected_result[i] = sum;
  }

  ASSERT_EQ(result_vector_, expected_result);
}

TEST(sedova_o_vertical_ribbon_scheme_seq, test_task_run) {
  int rows_ = 2000;
  int cols_ = 2000;
  std::vector<int> input_matrix_(rows_ * cols_);
  std::vector<int> input_vector_(cols_);
  std::vector<int> result_vector_(rows_, 0);

  for (int cols = 0; cols < cols_; ++cols) {
    for (int rows = 0; rows < rows_; ++rows) {
      input_matrix_[rows + cols * rows_] = (rows + cols * rows_) % 100;
    }
  }
  for (int i = 0; i < cols_; ++i) {
    input_vector_[i] = i % 50;
  }

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_.data()));
  taskDataSeq->inputs_count.emplace_back(input_matrix_.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector_.data()));
  taskDataSeq->inputs_count.emplace_back(input_vector_.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_vector_.data()));
  taskDataSeq->outputs_count.emplace_back(result_vector_.size());

  auto taskSequential = std::make_shared<sedova_o_vertical_ribbon_scheme_seq::Sequential>(taskDataSeq);
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

  std::vector<int> expected_result(rows_, 0);
  for (int i = 0; i < rows_; ++i) {
    int sum = 0;
    for (int j = 0; j < cols_; ++j) {
      sum += input_matrix_[i + j * rows_] * input_vector_[j];
    }
    expected_result[i] = sum;
  }
  ASSERT_EQ(result_vector_, expected_result);
}