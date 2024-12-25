#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>
#include <algorithm>

#include "mpi/sedova_o_multiplication_matrices_ccs/include/ops_mpi.hpp"

namespace sedova_o_multiplication_matrices_ccs_mpi {

std::vector<std::vector<double>> gen_rand_matrix(int rows, int cols, int non_zero_count) {
  // Initialize a matrix filled with zeros
  std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols, 0.0));

  // Create a list of all possible positions (row, col)
  std::vector<std::pair<int, int>> positions;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      positions.emplace_back(r, c);
    }
  }

  // Shuffle the positions to randomize them
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(positions.begin(), positions.end(), gen);

  // Assign random values to the first 'non_zero_count' positions
  std::uniform_real_distribution<> value_dist(-10.0, 10.0);
  for (size_t i = 0; i < non_zero_count && i < positions.size(); ++i) {
    int r = positions[i].first;
    int c = positions[i].second;
    matrix[r][c] = value_dist(gen);
  }

  return matrix;
}


std::vector<std::vector<double>> multiply_matrices(const std::vector<std::vector<double>>& A,
                                                   const std::vector<std::vector<double>>& B) {
  int rows_A = A.size();
  int cols_A = A[0].size();
  int cols_B = B[0].size();

  // Initialize the result matrix with zeros
  std::vector<std::vector<double>> result(rows_A, std::vector<double>(cols_B, 0.0));

  // Iterate over each row of A
  for (int i = 0; i < rows_A; ++i) {
    // Pre-compute the row of A that will be used in this iteration
    const std::vector<double>& rowA = A[i];

    // Iterate over each column of B
    for (int j = 0; j < cols_B; ++j) {
      double sum = 0.0;  // Initialize sum for this cell

      // Iterate over each element in the row of A and column of B
      for (int k = 0; k < cols_A; ++k) {
        sum += rowA[k] * B[k][j];  // Accumulate the product
      }

      result[i][j] = sum;  // Store the computed sum in the result matrix
    }
  }

  return result;
}

void func_test_template(const std::vector<std::vector<double>>& A_, const std::vector<std::vector<double>>& B_) {
  boost::mpi::communicator world;
  // Get dimensions of matrices A and B
  int rows_A = A_.size();
  int cols_A = (rows_A > 0) ? A_[0].size() : 0;  // Safe access
  int rows_B = B_.size();
  int cols_B = (rows_B > 0) ? B_[0].size() : 0;  // Safe access

  // Convert matrices A and B to CCS format
  std::vector<double> A_val, B_val;
  std::vector<int> A_row_ind, B_row_ind;
  std::vector<int> A_col_ptr, B_col_ptr;
  // Prepare expected result C on rank 0
  std::vector<double> exp_C_val;
  std::vector<int> exp_C_row_ind;
  std::vector<int> exp_C_col_ptr;

  if (world.rank() == 0) {
    auto exp_C = multiply_matrices(A_, B_);
    convertMatrixToCCS(exp_C, exp_C.size(), exp_C.empty() ? 0 : exp_C[0].size(), exp_C_val, exp_C_row_ind,
                       exp_C_col_ptr);
  }
  convertMatrixToCCS(A_, rows_A, cols_A, A_val, A_row_ind, A_col_ptr);
  convertMatrixToCCS(B_, rows_B, cols_B, B_val, B_row_ind, B_col_ptr);
  // Define C_val, C_row_ind, and C_col_ptr to hold results from the multiplication.
  std::vector<double> C_val;
  std::vector<int> C_row_ind;
  std::vector<int> C_col_ptr;

  // Prepare task data for matrix multiplication
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows_A));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols_A));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&rows_B));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cols_B));
  task_data->inputs_count.emplace_back(1);

  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_val.data()));
    task_data->inputs_count.emplace_back(A_val.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_row_ind.data()));
    task_data->inputs_count.emplace_back(A_row_ind.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(A_col_ptr.data()));
    task_data->inputs_count.emplace_back(A_col_ptr.size());

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_val.data()));
    task_data->inputs_count.emplace_back(B_val.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_row_ind.data()));
    task_data->inputs_count.emplace_back(B_row_ind.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(B_col_ptr.data()));
    task_data->inputs_count.emplace_back(B_col_ptr.size());

    // Resize expected output C
    C_val.resize(exp_C_val.size());
    C_row_ind.resize(exp_C_row_ind.size());
    C_col_ptr.resize(exp_C_col_ptr.size());

    task_data->outputs.reserve(3);  // Reserve space for outputs
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_val.data()));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_row_ind.data()));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_col_ptr.data()));
  }

 // Create and run the matrix multiplication task
  MatrixMultiplicationCCS task(task_data);

  // Validate and execute the multiplication if valid
  bool validation = task.validation();

  // Broadcast validation status to all processes
  boost::mpi::broadcast(world, validation, 0);

  if (validation) {
    task.pre_processing();
    task.run();
    task.post_processing();

    // Validate results on rank 0
    if (world.rank() == 0) {
      ASSERT_EQ(exp_C_val, C_val);
      ASSERT_EQ(exp_C_row_ind, C_row_ind);
      ASSERT_EQ(exp_C_col_ptr, C_col_ptr);
    }
  }
}

}  // namespace sedova_o_multiplication_matrices_ccs_mpi

TEST(sedova_o_multiplication_matrices_ccs_mpi, IdentityMatrix) {
  std::vector<std::vector<double>> I = {{1.0, 0.0}, {0.0, 1.0}};
  std::vector<std::vector<double>> A = {{2.0, 3.0}, {4.0, 5.0}};
  sedova_o_multiplication_matrices_ccs_mpi::func_test_template(I, A);
}

// Test for zero matrix multiplication
TEST(sedova_o_multiplication_matrices_ccs_mpi, ZeroMatrix) {
  std::vector<std::vector<double>> Z = {{0.0, 0.0}, {0.0, 0.0}};
  std::vector<std::vector<double>> A = {{2.0, 3.0}, {4.0, 5.0}};
  sedova_o_multiplication_matrices_ccs_mpi::func_test_template(Z, A);
}

// Test for large random matrices
TEST(sedova_o_multiplication_matrices_ccs_mpi, RandomLargeMatrices) {
  auto A_ = sedova_o_multiplication_matrices_ccs_mpi::gen_rand_matrix(100, 50, 250);
  auto B_ = sedova_o_multiplication_matrices_ccs_mpi::gen_rand_matrix(50, 100, 500);
  sedova_o_multiplication_matrices_ccs_mpi::func_test_template(A_, B_);
}

// Test for rectangular matrices with more rows than columns
TEST(sedova_o_multiplication_matrices_ccs_mpi, RectangularMoreRows) {
  std::vector<std::vector<double>> A_ = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  std::vector<std::vector<double>> B_ = {{7.0}, {8.0}};
  sedova_o_multiplication_matrices_ccs_mpi::func_test_template(A_, B_);
}

// Test for rectangular matrices with more columns than rows
TEST(sedova_o_multiplication_matrices_ccs_mpi, RectangularMoreColumns) {
  std::vector<std::vector<double>> A_ = {{1.0}, {2.0}};
  std::vector<std::vector<double>> B_ = {{3.0, 4.0}, {5.0, 6.0}};
  sedova_o_multiplication_matrices_ccs_mpi::func_test_template(A_, B_);
}

// Test for multiplying a matrix by a scalar (identity effect)
TEST(sedova_o_multiplication_matrices_ccs_mpi, ScalarMultiplication) {
  std::vector<std::vector<double>> A_ = {{1.0}, {2.0}};
  std::vector<std::vector<double>> B_ = {{3.0}};
  std::vector<std::vector<double>> expected_C = {{3.0}, {6.0}};
  sedova_o_multiplication_matrices_ccs_mpi::func_test_template(A_, B_);
}

// Validate test for small matrices with known output
TEST(sedova_o_multiplication_matrices_ccs_mpi, ValidateSmallMatrices) {
  std::vector<std::vector<double>> A_ = {{1.0}};
  std::vector<std::vector<double>> B_ = {{2.0}};
  std::vector<std::vector<double>> expected_C = {{2.0}};
  sedova_o_multiplication_matrices_ccs_mpi::func_test_template(A_, B_);
}