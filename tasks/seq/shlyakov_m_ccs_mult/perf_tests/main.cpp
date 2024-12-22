// Copyright 2023 Nesterov Alexander

#include <gtest/gtest.h>

#include <vector>
#include <vector>
#include <memory>
#include <random>
#include <chrono>
#include <algorithm>

#include "core/perf/include/perf.hpp"
#include "seq/shlyakov_m_ccs_mult/include/ops_seq.hpp"

using namespace shlyakov_m_ccs_mult;

SparseMatrix matrix_to_ccs(const std::vector<std::vector<double>>& matrix) {
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

std::vector<std::vector<double>> create_sparse_matrix(int rows, int cols, double sparsity_level,
                                                      unsigned int seed = 0) {
  std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols, 1.0));

  int num_zeros = static_cast<int>(rows * cols * sparsity_level);

  std::vector<int> indices(rows * cols);
  for (int i = 0; i < indices.size(); ++i) {
    indices[i] = i;
  }

  std::mt19937 g(seed);

  std::shuffle(indices.begin(), indices.end(), g);

  for (int i = 0; i < num_zeros; ++i) {
    int index = indices[i];
    int row = index / cols;
    int col = index % cols;
    matrix[row][col] = 0.0;
  }

  return matrix;
}

TEST(shlyakov_m_ccs_mult, test_pipeline_run) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
       
        int rows = 3000;
        int cols = 3000;
        double sparsity = 0.9;
        unsigned int seed1 = 123;
        unsigned int seed2 = 456;

         auto matrix_a = create_sparse_matrix(rows, cols, sparsity, seed1);
         auto matrix_b = create_sparse_matrix(rows, cols, sparsity, seed2);

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
       
      int rows = 3000;
      int cols = 3000;
      double sparsity = 0.9;
      unsigned int seed1 = 123;
      unsigned int seed2 = 456;

      auto matrix_a = create_sparse_matrix(rows, cols, sparsity, seed1);
      auto matrix_b = create_sparse_matrix(rows, cols, sparsity, seed2);

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