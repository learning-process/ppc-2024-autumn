#include "mpi/shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

bool shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::MatrixMultiplicationTaskSequential::pre_processing() {
  internal_order_test();

  std::vector<int> input_matrix_a_;
  std::vector<int> input_matrix_b_;

  int* matrix_a_data = reinterpret_cast<int*>(taskData->inputs[0]);
  int matrix_a_size = taskData->inputs_count[0];

  int* matrix_b_data = reinterpret_cast<int*>(taskData->inputs[1]);
  int matrix_b_size = taskData->inputs_count[1];

  input_matrix_a_.assign(matrix_a_data, matrix_a_data + matrix_a_size);
  input_matrix_b_.assign(matrix_b_data, matrix_b_data + matrix_b_size);

  num_rows_a_ = *reinterpret_cast<int*>(taskData->inputs[2]);
  num_cols_a_ = *reinterpret_cast<int*>(taskData->inputs[3]);
  num_cols_b_ = *reinterpret_cast<int*>(taskData->inputs[4]);

  int result_size = taskData->outputs_count[0];
  result_vector_.resize(result_size, 0);

  matA = Matrix(input_matrix_a_, num_rows_a_, num_cols_a_);
  matB = Matrix(input_matrix_b_, num_cols_a_, num_cols_b_);

  return true;
}

bool shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::MatrixMultiplicationTaskSequential::validation() {
  internal_order_test();

  int num_r_a_ = *reinterpret_cast<int*>(taskData->inputs[2]);
  int num_c_a_ = *reinterpret_cast<int*>(taskData->inputs[3]);
  int num_c_b_ = *reinterpret_cast<int*>(taskData->inputs[4]);

  return (taskData->inputs_count.size() > 3 && !taskData->outputs_count.empty() &&
          (num_r_a_ * num_c_a_ * num_c_b_ != 0));
}

bool shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::MatrixMultiplicationTaskSequential::run() {
  internal_order_test();
  result_vector_.resize(num_rows_a_ * num_cols_b_, 0);

  for (int i = 0; i < num_rows_a_; ++i) {
    for (int j = 0; j < num_cols_b_; ++j) {
      int sum = 0;

      for (int k = 0; k < num_cols_a_; ++k) {
        sum += matA.matrix_[i * num_cols_a_ + k] * matB.matrix_[k * num_cols_b_ + j];
      }

      result_vector_[i * num_cols_b_ + j] = sum;
    }
  }

  return true;
}

bool shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::MatrixMultiplicationTaskSequential::post_processing() {
  internal_order_test();

  int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(result_vector_.begin(), result_vector_.end(), output_data);

  return true;
}

void shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::get_indexes(int num_rows_a_, int num_rows_b_,
                                                                     std::vector<int>& indexesA,
                                                                     std::vector<int>& indexesB) {
  indexesA.resize(num_rows_a_ * num_rows_b_);
  indexesB.resize(num_rows_a_ * num_rows_b_);
  for (int i = 0; i < num_rows_a_; i++) {
    for (int j = 0; j < num_rows_b_; j++) {
      int index = i * num_rows_b_ + j;
      indexesA[index] = i;
      indexesB[index] = j;
    }
  }
}

void shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::calculate(int rows, int cols, int num_proc,
                                                                   std::vector<int>& sizes, std::vector<int>& displs) {
  sizes.resize(num_proc, 0);
  displs.resize(num_proc, 0);

  if (num_proc > rows) {
    for (int i = 0; i < rows; ++i) {
      sizes[i] = cols;
      displs[i] = i * cols;
    }
  } else {
    int a = rows / num_proc;
    int b = rows % num_proc;

    int offset = 0;
    for (int i = 0; i < num_proc; ++i) {
      if (b-- > 0) {
        sizes[i] = (a + 1) * cols;
      } else {
        sizes[i] = a * cols;
      }
      displs[i] = offset;
      offset += sizes[i];
    }
  }
}

bool shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::MatrixMultiplicationTaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    std::vector<int> input_matrix_a_;
    std::vector<int> input_matrix_b_;

    int* matrix_a_data = reinterpret_cast<int*>(taskData->inputs[0]);
    int matrix_a_size = taskData->inputs_count[0];

    int* matrix_b_data = reinterpret_cast<int*>(taskData->inputs[1]);
    int matrix_b_size = taskData->inputs_count[1];

    input_matrix_a_.assign(matrix_a_data, matrix_a_data + matrix_a_size);
    input_matrix_b_.assign(matrix_b_data, matrix_b_data + matrix_b_size);

    num_rows_a_ = *reinterpret_cast<int*>(taskData->inputs[2]);
    num_cols_a_ = *reinterpret_cast<int*>(taskData->inputs[3]);
    num_cols_b_ = *reinterpret_cast<int*>(taskData->inputs[4]);

    int result_size = taskData->outputs_count[0];
    result_vector_.resize(result_size, 0);

    matA = Matrix(input_matrix_a_, num_rows_a_, num_cols_a_);
    matB = Matrix(input_matrix_b_, num_cols_a_, num_cols_b_);

    shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::get_indexes(num_rows_a_, num_cols_b_, indexesA_, indexesB_);

    shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::calculate(num_rows_a_ * num_cols_b_, 1, world.size(), sizes,
                                                                  displs);
  }

  return true;
}

bool shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::MatrixMultiplicationTaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    int num_r_a_ = *reinterpret_cast<int*>(taskData->inputs[2]);
    int num_c_a_ = *reinterpret_cast<int*>(taskData->inputs[3]);
    int num_c_b_ = *reinterpret_cast<int*>(taskData->inputs[4]);
    return (taskData->inputs_count.size() > 3 && !taskData->outputs_count.empty() &&
            (num_r_a_ * num_c_a_ * num_c_b_ != 0));
  }
  return true;
}

bool shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::MatrixMultiplicationTaskParallel::run() {
  internal_order_test();

  boost::mpi::broadcast(world, matA, 0);
  boost::mpi::broadcast(world, matB, 0);
  boost::mpi::broadcast(world, sizes, 0);
  boost::mpi::broadcast(world, displs, 0);

  int local_size = sizes[world.rank()];

  std::vector<int> local_indexes_a(local_size);
  std::vector<int> local_indexes_b(local_size);

  if (world.rank() == 0) {
    boost::mpi::scatterv(world, indexesA_.data(), sizes, displs, local_indexes_a.data(), local_size, 0);
    boost::mpi::scatterv(world, indexesB_.data(), sizes, displs, local_indexes_b.data(), local_size, 0);
  } else {
    boost::mpi::scatterv(world, local_indexes_a.data(), local_size, 0);
    boost::mpi::scatterv(world, local_indexes_b.data(), local_size, 0);
  }

  std::vector<int> local_result(local_size, 0);

  for (size_t k = 0; k < local_indexes_a.size(); ++k) {
    int i = local_indexes_a[k];
    int j = local_indexes_b[k];

    auto itA = matA.row_begin(i);
    auto itB = matB.col_begin(j);

    while (itA != matA.row_end(i) && itB != matB.col_end(j)) {
      local_result[k] += (*itA) * (*itB);
      ++itA;
      ++itB;
    }
  }

  if (world.rank() == 0) {
    boost::mpi::gatherv(world, local_result.data(), local_result.size(), result_vector_.data(), sizes, displs, 0);
  } else {
    boost::mpi::gatherv(world, local_result.data(), local_result.size(), 0);
  }

  return true;
}

bool shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi::MatrixMultiplicationTaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(result_vector_.begin(), result_vector_.end(), output_data);
  }

  return true;
}
