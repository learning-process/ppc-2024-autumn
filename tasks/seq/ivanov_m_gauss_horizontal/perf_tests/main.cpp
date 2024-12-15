// Copyright 2024 Ivanov Mike
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/ivanov_m_gauss_horizontal/include/ops_seq.hpp"

namespace ivanov_m_gauss_horizontal_seq {
std::vector<double> GenSolution(int size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> generator(-2, 2);
  std::vector<double> solution(size, 0);

  for (int i = 0; i < size; i++) {
    solution[i] = static_cast<double>(generator(gen));  // generating random coefficient in range [-2, 2]
  }
  return solution;
}

std::vector<double> GenMatrix(const std::vector<double> &solution) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> generator(-2, 2);
  std::vector<double> extended_matrix;
  int size = static_cast<int>(solution.size());

  // generate identity matrix
  for (int row = 0; row < size; row++) {
    for (int column = 0; column < size; column++) {
      if (row == column) {
        extended_matrix.push_back(1);
      } else {
        extended_matrix.push_back(0);
      }
    }
    extended_matrix.push_back(solution[row]);
  }

  // saturation left triangle
  for (int row = 1; row < size; row++) {
    for (int column = 0; column < row; column++) {
      extended_matrix[get_linear_index(row, column, size + 1)] +=
          extended_matrix[get_linear_index(row - 1, column, size + 1)];
    }
    extended_matrix[get_linear_index(row, size, size + 1)] +=
        extended_matrix[get_linear_index(row - 1, size, size + 1)];
  }

  // saturation of matrix by random numbers
  for (int row = size - 1; row > 0; row--) {
    int coef = generator(gen);
    for (int column = 0; column < size + 1; column++) {
      extended_matrix[get_linear_index(row - 1, column, size + 1)] +=
          coef * extended_matrix[get_linear_index(row, column, size + 1)];
    }
  }

  // saturation of matrix by random numbers
  for (int row = 0; row < size - 1; row++) {
    int coef = generator(gen);
    for (int column = 0; column < size + 1; column++) {
      extended_matrix[get_linear_index(row + 1, column, size + 1)] +=
          coef * extended_matrix[get_linear_index(row, column, size + 1)];
    }
  }

  return extended_matrix;
}
}  // namespace ivanov_m_gauss_horizontal_seq

TEST(ivanov_m_gauss_horizontal_seq_perf_test, test_pipeline_run) {
  // create data
  size_t n = 1000;
  std::vector<double> ans = ivanov_m_gauss_horizontal_seq::GenSolution(n);
  std::vector<double> matrix = ivanov_m_gauss_horizontal_seq::GenMatrix(ans);
  std::vector<double> out(n, 0);

  // create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // create Task
  auto testTaskSequential = std::make_shared<ivanov_m_gauss_horizontal_seq::TestTaskSequential>(taskDataSeq);

  // create Perf attributes
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
}

TEST(ivanov_m_gauss_horizontal_seq_perf_test, test_task_run) {
  // create data
  size_t n = 1000;
  std::vector<double> ans = ivanov_m_gauss_horizontal_seq::GenSolution(n);
  std::vector<double> matrix = ivanov_m_gauss_horizontal_seq::GenMatrix(ans);
  std::vector<double> out(n, 0);

  // create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // create Task
  auto testTaskSequential = std::make_shared<ivanov_m_gauss_horizontal_seq::TestTaskSequential>(taskDataSeq);

  // create Perf attributes
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
}