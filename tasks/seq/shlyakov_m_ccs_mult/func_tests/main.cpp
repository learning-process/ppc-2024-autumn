// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

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

static std::vector<std::vector<double>> matrix_multiply(const std::vector<std::vector<double>>& matrix_a,
                                                        const std::vector<std::vector<double>>& matrix_b) {
  int rows_a = matrix_a.size();
  int cols_a = matrix_a[0].size();
  int cols_b = matrix_b[0].size();

  std::vector<std::vector<double>> result(rows_a, std::vector<double>(cols_b, 0.0));

  for (int i = 0; i < rows_a; ++i) {
    for (int j = 0; j < cols_b; ++j) {
      for (int k = 0; k < cols_a; ++k) {
        result[i][j] += matrix_a[i][k] * matrix_b[k][j];
      }
    }
  }

  return result;
}

static bool are_ccs_matrices_equal(const SparseMatrix& a, const SparseMatrix& b, double tolerance = 1e-9) {
  if (a.values.size() != b.values.size() || a.row_indices.size() != b.row_indices.size() ||
      a.col_pointers.size() != b.col_pointers.size()) {
    return false;
  }

  for (size_t i = 0; i < a.values.size(); ++i) {
    if (std::abs(a.values[i] - b.values[i]) > tolerance) {
      return false;
    }
  }

  for (size_t i = 0; i < a.row_indices.size(); ++i) {
    if (a.row_indices[i] != b.row_indices[i]) {
      return false;
    }
  }

  for (size_t i = 0; i < a.col_pointers.size(); ++i) {
    if (a.col_pointers[i] != b.col_pointers[i]) {
      return false;
    }
  }
  return true;
}

TEST(shlyakov_m_ccs_mult_seq, empty_matrices_test) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  int rows = 0;
  int cols = 0;
  double sparsity = 0.0;

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

  taskData->inputs_count.push_back(ccs_matrix_a.values.size());
  taskData->inputs_count.push_back(ccs_matrix_a.row_indices.size());
  taskData->inputs_count.push_back(ccs_matrix_a.col_pointers.size() - 1);
  taskData->inputs_count.push_back(ccs_matrix_b.values.size());
  taskData->inputs_count.push_back(ccs_matrix_b.row_indices.size());
  taskData->inputs_count.push_back(ccs_matrix_b.col_pointers.size() - 1);

  TestTaskSequential task(taskData);

  ASSERT_FALSE(task.validation());
}

TEST(shlyakov_m_ccs_mult_seq, square_matrices_test1) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  int rows = 10;
  int cols = 10;
  double sparsity = 0.5;

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

  taskData->inputs_count.push_back(ccs_matrix_a.values.size());
  taskData->inputs_count.push_back(ccs_matrix_a.row_indices.size());
  taskData->inputs_count.push_back(cols);
  taskData->inputs_count.push_back(ccs_matrix_b.values.size());
  taskData->inputs_count.push_back(ccs_matrix_b.row_indices.size());
  taskData->inputs_count.push_back(rows);

  TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  std::vector<std::vector<double>> expected_result_matrix = matrix_multiply(matrix_a, matrix_b);
  SparseMatrix expected_result = matrix_to_ccs(expected_result_matrix);

  SparseMatrix actual_result;
  const auto* values_ptr = reinterpret_cast<const double*>(taskData->outputs[0]);
  unsigned int values_size = taskData->outputs_count[0] / sizeof(double);

  const int* row_indices_ptr = reinterpret_cast<const int*>(taskData->outputs[1]);
  unsigned int row_indices_size = taskData->outputs_count[1] / sizeof(int);

  const int* col_pointers_ptr = reinterpret_cast<const int*>(taskData->outputs[2]);
  unsigned int col_pointers_size = taskData->outputs_count[2] / sizeof(int);

  actual_result.values.assign(values_ptr, values_ptr + values_size);
  actual_result.row_indices.assign(row_indices_ptr, row_indices_ptr + row_indices_size);
  actual_result.col_pointers.assign(col_pointers_ptr, col_pointers_ptr + col_pointers_size);

  ASSERT_TRUE(are_ccs_matrices_equal(expected_result, actual_result, 1e-9));
}

TEST(shlyakov_m_ccs_mult_seq, square_matrices_test2) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  int rows = 3;
  int cols = 3;
  double sparsity = 0.5;

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

  taskData->inputs_count.push_back(ccs_matrix_a.values.size());
  taskData->inputs_count.push_back(ccs_matrix_a.row_indices.size());
  taskData->inputs_count.push_back(ccs_matrix_a.col_pointers.size() - 1);
  taskData->inputs_count.push_back(ccs_matrix_b.values.size());
  taskData->inputs_count.push_back(ccs_matrix_b.row_indices.size());
  taskData->inputs_count.push_back(ccs_matrix_b.col_pointers.size() - 1);

  TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  std::vector<std::vector<double>> expected_result_matrix = matrix_multiply(matrix_a, matrix_b);
  SparseMatrix expected_result = matrix_to_ccs(expected_result_matrix);

  SparseMatrix actual_result;
  const auto* values_ptr = reinterpret_cast<const double*>(taskData->outputs[0]);
  unsigned int values_size = taskData->outputs_count[0] / sizeof(double);

  const int* row_indices_ptr = reinterpret_cast<const int*>(taskData->outputs[1]);
  unsigned int row_indices_size = taskData->outputs_count[1] / sizeof(int);

  const int* col_pointers_ptr = reinterpret_cast<const int*>(taskData->outputs[2]);
  unsigned int col_pointers_size = taskData->outputs_count[2] / sizeof(int);

  actual_result.values.assign(values_ptr, values_ptr + values_size);
  actual_result.row_indices.assign(row_indices_ptr, row_indices_ptr + row_indices_size);
  actual_result.col_pointers.assign(col_pointers_ptr, col_pointers_ptr + col_pointers_size);

  ASSERT_TRUE(are_ccs_matrices_equal(expected_result, actual_result, 1e-9));
}

TEST(shlyakov_m_ccs_mult_seq, rectangular_matrices_test1) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  int rows_a = 2;
  int cols_a = 3;
  int rows_b = 3;
  int cols_b = 4;
  double sparsity = 0.4;

  auto matrix_a = generate_random_sparse_matrix(rows_a, cols_a, sparsity);
  auto matrix_b = generate_random_sparse_matrix(rows_b, cols_b, sparsity);

  SparseMatrix ccs_matrix_a = matrix_to_ccs(matrix_a);
  SparseMatrix ccs_matrix_b = matrix_to_ccs(matrix_b);

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(ccs_matrix_a.values.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(ccs_matrix_a.row_indices.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(ccs_matrix_a.col_pointers.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(ccs_matrix_b.values.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(ccs_matrix_b.row_indices.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(ccs_matrix_b.col_pointers.data()));

  taskData->inputs_count.push_back(ccs_matrix_a.values.size());
  taskData->inputs_count.push_back(ccs_matrix_a.row_indices.size());
  taskData->inputs_count.push_back(ccs_matrix_a.col_pointers.size() - 1);
  taskData->inputs_count.push_back(ccs_matrix_b.values.size());
  taskData->inputs_count.push_back(ccs_matrix_b.row_indices.size());
  taskData->inputs_count.push_back(ccs_matrix_b.col_pointers.size() - 1);

  TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  std::vector<std::vector<double>> expected_result_matrix = matrix_multiply(matrix_a, matrix_b);
  SparseMatrix expected_result = matrix_to_ccs(expected_result_matrix);

  SparseMatrix actual_result;
  const auto* values_ptr = reinterpret_cast<const double*>(taskData->outputs[0]);
  unsigned int values_size = taskData->outputs_count[0] / sizeof(double);

  const int* row_indices_ptr = reinterpret_cast<const int*>(taskData->outputs[1]);
  unsigned int row_indices_size = taskData->outputs_count[1] / sizeof(int);

  const int* col_pointers_ptr = reinterpret_cast<const int*>(taskData->outputs[2]);
  unsigned int col_pointers_size = taskData->outputs_count[2] / sizeof(int);

  actual_result.values.assign(values_ptr, values_ptr + values_size);
  actual_result.row_indices.assign(row_indices_ptr, row_indices_ptr + row_indices_size);
  actual_result.col_pointers.assign(col_pointers_ptr, col_pointers_ptr + col_pointers_size);

  ASSERT_TRUE(are_ccs_matrices_equal(expected_result, actual_result, 1e-9));
}

TEST(shlyakov_m_ccs_mult_seq, rectangular_matrices_test2) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  int rows_a = 5;
  int cols_a = 10;
  int rows_b = 10;
  int cols_b = 5;
  double sparsity = 0.7;

  auto matrix_a = generate_random_sparse_matrix(rows_a, cols_a, sparsity);
  auto matrix_b = generate_random_sparse_matrix(rows_b, cols_b, sparsity);

  SparseMatrix ccs_matrix_a = matrix_to_ccs(matrix_a);
  SparseMatrix ccs_matrix_b = matrix_to_ccs(matrix_b);

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(ccs_matrix_a.values.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(ccs_matrix_a.row_indices.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(ccs_matrix_a.col_pointers.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(ccs_matrix_b.values.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(ccs_matrix_b.row_indices.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(ccs_matrix_b.col_pointers.data()));

  taskData->inputs_count.push_back(ccs_matrix_a.values.size());
  taskData->inputs_count.push_back(ccs_matrix_a.row_indices.size());
  taskData->inputs_count.push_back(ccs_matrix_a.col_pointers.size() - 1);
  taskData->inputs_count.push_back(ccs_matrix_b.values.size());
  taskData->inputs_count.push_back(ccs_matrix_b.row_indices.size());
  taskData->inputs_count.push_back(ccs_matrix_b.col_pointers.size() - 1);

  TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  std::vector<std::vector<double>> expected_result_matrix = matrix_multiply(matrix_a, matrix_b);
  SparseMatrix expected_result = matrix_to_ccs(expected_result_matrix);

  SparseMatrix actual_result;
  const auto* values_ptr = reinterpret_cast<const double*>(taskData->outputs[0]);
  unsigned int values_size = taskData->outputs_count[0] / sizeof(double);

  const int* row_indices_ptr = reinterpret_cast<const int*>(taskData->outputs[1]);
  unsigned int row_indices_size = taskData->outputs_count[1] / sizeof(int);

  const int* col_pointers_ptr = reinterpret_cast<const int*>(taskData->outputs[2]);
  unsigned int col_pointers_size = taskData->outputs_count[2] / sizeof(int);

  actual_result.values.assign(values_ptr, values_ptr + values_size);
  actual_result.row_indices.assign(row_indices_ptr, row_indices_ptr + row_indices_size);
  actual_result.col_pointers.assign(col_pointers_ptr, col_pointers_ptr + col_pointers_size);

  ASSERT_TRUE(are_ccs_matrices_equal(expected_result, actual_result, 1e-9));
}

TEST(shlyakov_m_ccs_mult_seq, single_element_matrix_test) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  int rows = 1;
  int cols = 1;
  double sparsity = 1.0;

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

  taskData->inputs_count.push_back(ccs_matrix_a.values.size());
  taskData->inputs_count.push_back(ccs_matrix_a.row_indices.size());
  taskData->inputs_count.push_back(ccs_matrix_a.col_pointers.size() - 1);
  taskData->inputs_count.push_back(ccs_matrix_b.values.size());
  taskData->inputs_count.push_back(ccs_matrix_b.row_indices.size());
  taskData->inputs_count.push_back(ccs_matrix_b.col_pointers.size() - 1);

  TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  std::vector<std::vector<double>> expected_result_matrix = matrix_multiply(matrix_a, matrix_b);
  SparseMatrix expected_result = matrix_to_ccs(expected_result_matrix);

  SparseMatrix actual_result;
  const auto* values_ptr = reinterpret_cast<const double*>(taskData->outputs[0]);
  unsigned int values_size = taskData->outputs_count[0] / sizeof(double);

  const int* row_indices_ptr = reinterpret_cast<const int*>(taskData->outputs[1]);
  unsigned int row_indices_size = taskData->outputs_count[1] / sizeof(int);

  const int* col_pointers_ptr = reinterpret_cast<const int*>(taskData->outputs[2]);
  unsigned int col_pointers_size = taskData->outputs_count[2] / sizeof(int);

  actual_result.values.assign(values_ptr, values_ptr + values_size);
  actual_result.row_indices.assign(row_indices_ptr, row_indices_ptr + row_indices_size);
  actual_result.col_pointers.assign(col_pointers_ptr, col_pointers_ptr + col_pointers_size);

  ASSERT_TRUE(are_ccs_matrices_equal(expected_result, actual_result, 1e-9));
}

TEST(shlyakov_m_ccs_mult_seq, sparse_matrices_test) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  int rows = 10;
  int cols = 10;
  double sparsity = 0.9;

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

  taskData->inputs_count.push_back(ccs_matrix_a.values.size());
  taskData->inputs_count.push_back(ccs_matrix_a.row_indices.size());
  taskData->inputs_count.push_back(ccs_matrix_a.col_pointers.size() - 1);
  taskData->inputs_count.push_back(ccs_matrix_b.values.size());
  taskData->inputs_count.push_back(ccs_matrix_b.row_indices.size());
  taskData->inputs_count.push_back(ccs_matrix_b.col_pointers.size() - 1);

  TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  std::vector<std::vector<double>> expected_result_matrix = matrix_multiply(matrix_a, matrix_b);
  SparseMatrix expected_result = matrix_to_ccs(expected_result_matrix);

  SparseMatrix actual_result;
  const auto* values_ptr = reinterpret_cast<const double*>(taskData->outputs[0]);
  unsigned int values_size = taskData->outputs_count[0] / sizeof(double);

  const int* row_indices_ptr = reinterpret_cast<const int*>(taskData->outputs[1]);
  unsigned int row_indices_size = taskData->outputs_count[1] / sizeof(int);

  const int* col_pointers_ptr = reinterpret_cast<const int*>(taskData->outputs[2]);
  unsigned int col_pointers_size = taskData->outputs_count[2] / sizeof(int);

  actual_result.values.assign(values_ptr, values_ptr + values_size);
  actual_result.row_indices.assign(row_indices_ptr, row_indices_ptr + row_indices_size);
  actual_result.col_pointers.assign(col_pointers_ptr, col_pointers_ptr + col_pointers_size);

  ASSERT_TRUE(are_ccs_matrices_equal(expected_result, actual_result, 1e-9));
}

TEST(shlyakov_m_ccs_mult_seq, all_zeros_matrices_test) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  int rows = 5;
  int cols = 5;
  double sparsity = 0.0;

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

  taskData->inputs_count.push_back(ccs_matrix_a.values.size());
  taskData->inputs_count.push_back(ccs_matrix_a.row_indices.size());
  taskData->inputs_count.push_back(ccs_matrix_a.col_pointers.size() - 1);
  taskData->inputs_count.push_back(ccs_matrix_b.values.size());
  taskData->inputs_count.push_back(ccs_matrix_b.row_indices.size());
  taskData->inputs_count.push_back(ccs_matrix_b.col_pointers.size() - 1);

  TestTaskSequential task(taskData);
  ASSERT_FALSE(task.validation());
}
