// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>
#include <random>
#include <algorithm>

#include "core/perf/include/perf.hpp"

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
  for (int i = 0; i < non_zero_count; ++i) {
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
}  // namespace sedova_o_multiplication_matrices_ccs_mpi

TEST(sedova_o_multiplication_matrices_ccs_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int size = 256;
  int elements = 7000;
  std::vector<std::vector<double>> A_;
  std::vector<std::vector<double>> B_;

  if (world.rank() == 0) {
    A_ = sedova_o_multiplication_matrices_ccs_mpi::gen_rand_matrix(size, size, elements);
    B_ = sedova_o_multiplication_matrices_ccs_mpi::gen_rand_matrix(size, size, elements);
  }

  boost::mpi::broadcast(world, A_, 0);
  boost::mpi::broadcast(world, B_, 0);

  int rows_A = A_.size();
  int cols_A = (rows_A > 0) ? A_[0].size() : 0;
  int rows_B = B_.size();
  int cols_B = (rows_B > 0) ? B_[0].size() : 0;
  if (cols_A != rows_B) {
    throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
  }
  std::vector<double> A_val, B_val;
  std::vector<int> A_row_ind, B_row_ind;
  std::vector<int> A_col_ptr, B_col_ptr;

  sedova_o_multiplication_matrices_ccs_mpi::convertMatrixToCCS(A_, rows_A, cols_A, A_val, A_row_ind, A_col_ptr);
  sedova_o_multiplication_matrices_ccs_mpi::convertMatrixToCCS(B_, rows_B, cols_B, B_val, B_row_ind, B_col_ptr);

  std::vector<double> exp_C_val;
  std::vector<int> exp_C_row_ind;
  std::vector<int> exp_C_col_ptr;

  if (world.rank() == 0) {
    auto exp_C = sedova_o_multiplication_matrices_ccs_mpi::multiply_matrices(A_, B_);
    sedova_o_multiplication_matrices_ccs_mpi::convertMatrixToCCS(
        exp_C, exp_C.size(), exp_C.empty() ? 0 : exp_C[0].size(), exp_C_val, exp_C_row_ind,
                       exp_C_col_ptr);
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.reserve(8);        
  task_data->inputs_count.reserve(8);  
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

    exp_C_val.resize(exp_C_val.size());
    exp_C_row_ind.resize(exp_C_row_ind.size());
    exp_C_col_ptr.resize(exp_C_col_ptr.size());

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(exp_C_val.data()));
    task_data->outputs_count.emplace_back(exp_C_val.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(exp_C_row_ind.data()));
    task_data->outputs_count.emplace_back(exp_C_row_ind.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(exp_C_col_ptr.data()));
    task_data->outputs_count.emplace_back(exp_C_col_ptr.size());
  }

  auto task = std::make_shared<sedova_o_multiplication_matrices_ccs_mpi::MatrixMultiplicationCCS>(task_data);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

    if (world.rank() == 0) {
    std::vector<double> C_val;
    std::vector<int> C_row_ind;
    std::vector<int> C_col_ptr;
    ASSERT_EQ(exp_C_val, C_val);
    ASSERT_EQ(exp_C_row_ind, C_row_ind);
    ASSERT_EQ(exp_C_col_ptr, C_col_ptr);
  }
}

TEST(sedova_o_multiplication_matrices_ccs_mpi, test_task_run) {
  boost::mpi::communicator world;
  int size = 256;
  int elements = 7000;
  std::vector<std::vector<double>> A_;
  std::vector<std::vector<double>> B_;

  if (world.rank() == 0) {
    A_ = sedova_o_multiplication_matrices_ccs_mpi::gen_rand_matrix(size, size, elements);
    B_ = sedova_o_multiplication_matrices_ccs_mpi::gen_rand_matrix(size, size, elements);
  }

  boost::mpi::broadcast(world, A_, 0);
  boost::mpi::broadcast(world, B_, 0);

  int rows_A = A_.size();
  int cols_A = (rows_A > 0) ? A_[0].size() : 0;
  int rows_B = B_.size();
  int cols_B = (rows_B > 0) ? B_[0].size() : 0;
  if (cols_A != rows_B) {
    throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
  }
  std::vector<double> A_val, B_val;
  std::vector<int> A_row_ind, B_row_ind;
  std::vector<int> A_col_ptr, B_col_ptr;

  sedova_o_multiplication_matrices_ccs_mpi::convertMatrixToCCS(A_, rows_A, cols_A, A_val, A_row_ind, A_col_ptr);
  sedova_o_multiplication_matrices_ccs_mpi::convertMatrixToCCS(B_, rows_B, cols_B, B_val, B_row_ind, B_col_ptr);

  std::vector<double> exp_C_val;
  std::vector<int> exp_C_row_ind;
  std::vector<int> exp_C_col_ptr;

  if (world.rank() == 0) {
    auto exp_C = sedova_o_multiplication_matrices_ccs_mpi::multiply_matrices(A_, B_);
    sedova_o_multiplication_matrices_ccs_mpi::convertMatrixToCCS(
        exp_C, exp_C.size(), exp_C.empty() ? 0 : exp_C[0].size(), exp_C_val, exp_C_row_ind, exp_C_col_ptr);
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.reserve(8);
  task_data->inputs_count.reserve(8);
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

    exp_C_val.resize(exp_C_val.size());
    exp_C_row_ind.resize(exp_C_row_ind.size());
    exp_C_col_ptr.resize(exp_C_col_ptr.size());

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(exp_C_val.data()));
    task_data->outputs_count.emplace_back(exp_C_val.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(exp_C_row_ind.data()));
    task_data->outputs_count.emplace_back(exp_C_row_ind.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(exp_C_col_ptr.data()));
    task_data->outputs_count.emplace_back(exp_C_col_ptr.size());
  }

  auto task = std::make_shared<sedova_o_multiplication_matrices_ccs_mpi::MatrixMultiplicationCCS>(task_data);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->task_run(perfAttr, perfResults);

    if (world.rank() == 0) {
    std::vector<double> C_val;
    std::vector<int> C_row_ind;
    std::vector<int> C_col_ptr;
    ASSERT_EQ(exp_C_val, C_val);
    ASSERT_EQ(exp_C_row_ind, C_row_ind);
    ASSERT_EQ(exp_C_col_ptr, C_col_ptr);
  }
}