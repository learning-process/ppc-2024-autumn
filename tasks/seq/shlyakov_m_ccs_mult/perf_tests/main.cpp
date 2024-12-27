// Copyright 2023 Nesterov Alexander

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/shlyakov_m_ccs_mult/include/ops_seq.hpp"

using namespace shlyakov_m_ccs_mult;

static SparseMatrix matrix_to_ccs(const std::vector<std::vector<double>>& matrix) {
  SparseMatrix ccs_matrix;
  int rows = matrix.size();

  if (rows == 0) {
    ccs_matrix.col_pointers.push_back(0);
    return ccs_matrix;
  }

  int cols = matrix[0].size();

  ccs_matrix.col_pointers.push_back(0);

  for (int col = 0; col < cols; ++col) {
    for (int row = 0; row < rows; ++row) {
      if (matrix[row][col] != 0) {
        ccs_matrix.values.push_back(matrix[row][col]);
        ccs_matrix.row_indices.push_back(row);
      }
    }
    ccs_matrix.col_pointers.push_back(ccs_matrix.values.size());
  }

  return ccs_matrix;
}

static std::vector<std::vector<double>> generate_random_sparse_matrix(int rows, int cols, double density) {
  std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols, 0.0));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0, 1.0);
  std::normal_distribution<double> normal(0.0, 1.0);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if (dis(gen) < density) {
        matrix[i][j] = normal(gen);
      }
    }
  }
  return matrix;
}

TEST(shlyakov_m_ccs_mult, test_pipeline_run) {
  auto taskData = std::make_shared<ppc::core::TaskData>();

  int rows = 6000;
  int cols = 6000;
  double sparsity = 0.01;

  auto matrix_a = generate_random_sparse_matrix(rows, cols, sparsity);
  auto matrix_b = generate_random_sparse_matrix(rows, cols, sparsity);

  SparseMatrix ccs_matrix_a = matrix_to_ccs(matrix_a);
  SparseMatrix ccs_matrix_b = matrix_to_ccs(matrix_b);

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(ccs_matrix_a.values.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(ccs_matrix_a.row_indices.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(ccs_matrix_a.col_pointers.data()));

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(ccs_matrix_b.values.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(ccs_matrix_b.row_indices.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(ccs_matrix_b.col_pointers.data()));

  taskData->inputs_count.push_back(static_cast<unsigned int>(ccs_matrix_a.values.size()));
  taskData->inputs_count.push_back(static_cast<unsigned int>(ccs_matrix_a.row_indices.size()));
  taskData->inputs_count.push_back(cols);

  taskData->inputs_count.push_back(static_cast<unsigned int>(ccs_matrix_b.values.size()));
  taskData->inputs_count.push_back(static_cast<unsigned int>(ccs_matrix_b.row_indices.size()));
  taskData->inputs_count.push_back(rows);

  TestTaskSequential seqTask(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(std::make_shared<TestTaskSequential>(taskData));
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(shlyakov_m_ccs_mult, test_task_run) {
  auto taskData = std::make_shared<ppc::core::TaskData>();

  int rows = 6000;
  int cols = 6000;
  double sparsity = 0.01;

  auto matrix_a = generate_random_sparse_matrix(rows, cols, sparsity);
  auto matrix_b = generate_random_sparse_matrix(rows, cols, sparsity);

  SparseMatrix ccs_matrix_a = matrix_to_ccs(matrix_a);
  SparseMatrix ccs_matrix_b = matrix_to_ccs(matrix_b);

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(ccs_matrix_a.values.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(ccs_matrix_a.row_indices.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(ccs_matrix_a.col_pointers.data()));

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(ccs_matrix_b.values.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(ccs_matrix_b.row_indices.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(ccs_matrix_b.col_pointers.data()));

  taskData->inputs_count.push_back(static_cast<unsigned int>(ccs_matrix_a.values.size()));
  taskData->inputs_count.push_back(static_cast<unsigned int>(ccs_matrix_a.row_indices.size()));
  taskData->inputs_count.push_back(cols);

  taskData->inputs_count.push_back(static_cast<unsigned int>(ccs_matrix_b.values.size()));
  taskData->inputs_count.push_back(static_cast<unsigned int>(ccs_matrix_b.row_indices.size()));
  taskData->inputs_count.push_back(rows);

  TestTaskSequential seqTask(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(std::make_shared<TestTaskSequential>(taskData));
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}
