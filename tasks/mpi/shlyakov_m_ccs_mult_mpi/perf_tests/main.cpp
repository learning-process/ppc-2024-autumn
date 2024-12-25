// Copyright 2023 Nesterov Alexander

#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "mpi/shlyakov_m_ccs_mult_mpi/include/ops_mpi.hpp"

using namespace shlyakov_m_ccs_mult_mpi;

static SparseMatrix matrix_to_ccs(const std::vector<std::vector<double>>& matrix) {
  SparseMatrix ccs_matrix;
  size_t rows = matrix.size();

  if (rows == 0) {
    ccs_matrix.col_pointers.push_back(0);
    return ccs_matrix;
  }

  size_t cols = matrix[0].size();

  ccs_matrix.col_pointers.push_back(0);

  for (size_t col = 0; col < cols; ++col) {
    for (size_t row = 0; row < rows; ++row) {
      if (matrix[row][col] != 0) {
        ccs_matrix.values.push_back(matrix[row][col]);
        ccs_matrix.row_indices.push_back(static_cast<int>(row));
      }
    }
    ccs_matrix.col_pointers.push_back(static_cast<int>(ccs_matrix.values.size()));
  }

  return ccs_matrix;
}

std::vector<std::vector<double>> generate_random_sparse_matrix(int rows, int cols, double density) {
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

TEST(shlyakov_m_ccs_mult_mpi, test_pipeline_run) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;

  int rows = 6000;
  int cols = 6000;
  double density = 0.01;

  std::vector<std::vector<double>> dense_A = generate_random_sparse_matrix(rows, cols, density);
  std::vector<std::vector<double>> dense_B = generate_random_sparse_matrix(rows, cols, density);

  SparseMatrix A_ccs = matrix_to_ccs(dense_A);
  SparseMatrix B_ccs = matrix_to_ccs(dense_B);

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.values.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.row_indices.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.col_pointers.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.values.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.row_indices.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.col_pointers.data()));

  taskData->inputs_count.push_back(A_ccs.values.size());
  taskData->inputs_count.push_back(A_ccs.row_indices.size());
  taskData->inputs_count.push_back(A_ccs.col_pointers.size() - 1);
  taskData->inputs_count.push_back(B_ccs.values.size());
  taskData->inputs_count.push_back(B_ccs.row_indices.size());
  taskData->inputs_count.push_back(B_ccs.col_pointers.size() - 1);

  auto mpiTask = std::make_shared<shlyakov_m_ccs_mult_mpi::TestTaskMPI>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(mpiTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(shlyakov_m_ccs_mult_mpi, test_task_run) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;

  int rows = 6000;
  int cols = 6000;
  double density = 0.01;

  std::vector<std::vector<double>> dense_A = generate_random_sparse_matrix(rows, cols, density);
  std::vector<std::vector<double>> dense_B = generate_random_sparse_matrix(rows, cols, density);

  SparseMatrix A_ccs = matrix_to_ccs(dense_A);
  SparseMatrix B_ccs = matrix_to_ccs(dense_B);

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.values.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.row_indices.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(A_ccs.col_pointers.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.values.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.row_indices.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(B_ccs.col_pointers.data()));

  taskData->inputs_count.push_back(A_ccs.values.size());
  taskData->inputs_count.push_back(A_ccs.row_indices.size());
  taskData->inputs_count.push_back(A_ccs.col_pointers.size() - 1);
  taskData->inputs_count.push_back(B_ccs.values.size());
  taskData->inputs_count.push_back(B_ccs.row_indices.size());
  taskData->inputs_count.push_back(B_ccs.col_pointers.size() - 1);

  auto mpiTask = std::make_shared<shlyakov_m_ccs_mult_mpi::TestTaskMPI>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(mpiTask);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}
