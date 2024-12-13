#include <gtest/gtest.h>

#include <chrono>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/vasenkov_a_gauss_jordan_method_seq/include/ops_seq.hpp"



std::vector<double> generate_invertible_matrix(int size) {
  std::vector<double> matrix(size * (size + 1));
  std::random_device rd;
  std::mt19937 gen(rd());
  double lowerLimit = -100.0;
  double upperLimit = 100.0;
  std::uniform_real_distribution<> dist(lowerLimit, upperLimit);

  for (int i = 0; i < size; ++i) {
    double row_sum = 0.0;
    double diag = (i * (size + 1) + i);
    for (int j = 0; j < size + 1; ++j) {
      if (i != j) {
        matrix[i * (size + 1) + j] = dist(gen);
        row_sum += std::abs(matrix[i * (size + 1) + j]);
      }
    }
    matrix[diag] = row_sum + 1;
  }

  return matrix;
}


TEST(vasenkov_a_gauss_jordan_method_seq, pipeline_run) {
  int n = 50;
  std::vector<double> global_matrix = generate_invertible_matrix(n);
  std::vector<double> global_result(n * (n + 1));

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(global_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  taskDataSeq->outputs_count.emplace_back(global_result.size());

  auto taskSequential = std::make_shared<vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential>(taskDataSeq);
  EXPECT_TRUE(taskSequential->validation());
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
}

TEST(vasenkov_a_gauss_jordan_method_seq, task_run) {
  int n = 50;
  std::vector<double> global_matrix = generate_invertible_matrix(n);
  std::vector<double> global_result(n * (n + 1));

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(global_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  taskDataSeq->outputs_count.emplace_back(global_result.size());

  auto taskSequential = std::make_shared<vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential>(taskDataSeq);
  EXPECT_TRUE(taskSequential->validation());
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
}