#include "mpi/kalinin_d_matrix_mult_hor_a_vert_b/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

void kalinin_d_matrix_mult_hor_a_vert_b_mpi::compute_indexes(int _rows_a, int num_rows_b, std::vector<int>& indexesA,
                                                             std::vector<int>& indexesB) {
  indexesA.resize(_rows_a * num_rows_b);
  indexesB.resize(_rows_a * num_rows_b);

  for (int i = 0; i < _rows_a; i++) {
    for (int j = 0; j < num_rows_b; j++) {
      int index = i * num_rows_b + j;
      indexesA[index] = i;
      indexesB[index] = j;
    }
  }
}

void kalinin_d_matrix_mult_hor_a_vert_b_mpi::calculate(int rows, int columns, int num_proc, std::vector<int>& _sizes,
                                                       std::vector<int>& _displs) {
  _sizes.resize(num_proc, 0);
  _displs.resize(num_proc, 0);

  if (num_proc > rows) {
    for (int i = 0; i < rows; ++i) {
      _sizes[i] = columns;
      _displs[i] = i * columns;
    }
  } else {
    int rows_per_process = rows / num_proc;
    int remaining_rows = rows % num_proc;
    int offset = 0;

    for (int i = 0; i < num_proc; ++i) {
      if (remaining_rows-- > 0) {
        _sizes[i] = (rows_per_process + 1) * columns;
      } else {
        _sizes[i] = rows_per_process * columns;
      }

      _displs[i] = offset;
      offset += _sizes[i];
    }
  }
}

bool kalinin_d_matrix_mult_hor_a_vert_b_mpi::SequentialMatrixMultiplicationTask::pre_processing() {
  internal_order_test();

  std::vector<int> input_matrix_a;
  std::vector<int> input_matrix_b;

  int* matrix_a_data = reinterpret_cast<int*>(taskData->inputs[0]);
  int matrix_a_size = taskData->inputs_count[0];

  int* matrix_b_data = reinterpret_cast<int*>(taskData->inputs[1]);
  int matrix_b_size = taskData->inputs_count[1];

  input_matrix_a.assign(matrix_a_data, matrix_a_data + matrix_a_size);
  input_matrix_b.assign(matrix_b_data, matrix_b_data + matrix_b_size);

  _rows_a = *reinterpret_cast<int*>(taskData->inputs[2]);
  _columns_a = *reinterpret_cast<int*>(taskData->inputs[3]);
  _columns_b = *reinterpret_cast<int*>(taskData->inputs[4]);

  int result_size = taskData->outputs_count[0];
  _result_vector.resize(result_size, 0);

  _matrixA = Matrix(input_matrix_a, _rows_a, _columns_a);
  _matrixB = Matrix(input_matrix_b, _columns_a, _columns_b);

  return true;
}

bool kalinin_d_matrix_mult_hor_a_vert_b_mpi::SequentialMatrixMultiplicationTask::validation() {
  internal_order_test();

  int rows_a = *reinterpret_cast<int*>(taskData->inputs[2]);
  int columns_a = *reinterpret_cast<int*>(taskData->inputs[3]);
  int columns_b = *reinterpret_cast<int*>(taskData->inputs[4]);

  return (taskData->inputs_count.size() > 3 && !taskData->outputs_count.empty() &&
          (rows_a * columns_a * columns_b != 0));
}

bool kalinin_d_matrix_mult_hor_a_vert_b_mpi::SequentialMatrixMultiplicationTask::run() {
  internal_order_test();

  _result_vector.resize(_rows_a * _columns_b, 0);

  for (int i = 0; i < _rows_a; ++i) {
    for (int j = 0; j < _columns_b; ++j) {
      int sum = 0;

      for (int k = 0; k < _columns_a; ++k) {
        sum += _matrixA._matrixData[i * _columns_a + k] * _matrixB._matrixData[k * _columns_b + j];
      }

      _result_vector[i * _columns_b + j] = sum;
    }
  }

  return true;
}

bool kalinin_d_matrix_mult_hor_a_vert_b_mpi::SequentialMatrixMultiplicationTask::post_processing() {
  internal_order_test();

  int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(_result_vector.begin(), _result_vector.end(), output_data);

  return true;
}

bool kalinin_d_matrix_mult_hor_a_vert_b_mpi::ParallelMatrixMultiplicationTask::pre_processing() {
  internal_order_test();

  if (_world.rank() == 0) {
    std::vector<int> input_matrix_a;
    std::vector<int> input_matrix_b;

    int* matrix_a_data = reinterpret_cast<int*>(taskData->inputs[0]);
    int* matrix_b_data = reinterpret_cast<int*>(taskData->inputs[1]);

    int matrix_a_size = taskData->inputs_count[0];
    int matrix_b_size = taskData->inputs_count[1];

    input_matrix_a.assign(matrix_a_data, matrix_a_data + matrix_a_size);
    input_matrix_b.assign(matrix_b_data, matrix_b_data + matrix_b_size);

    _rows_a = *reinterpret_cast<int*>(taskData->inputs[2]);
    _columns_a = *reinterpret_cast<int*>(taskData->inputs[3]);
    _columns_b = *reinterpret_cast<int*>(taskData->inputs[4]);

    int result_size = taskData->outputs_count[0];

    _result_vector.resize(result_size, 0);

    _matrixA = Matrix(input_matrix_a, _rows_a, _columns_a);
    _matrixB = Matrix(input_matrix_b, _columns_a, _columns_b);

    kalinin_d_matrix_mult_hor_a_vert_b_mpi::compute_indexes(_rows_a, _columns_b, _indexesA, _indexesB);

    kalinin_d_matrix_mult_hor_a_vert_b_mpi::calculate(_rows_a * _columns_b, 1, _world.size(), _sizes, _displs);
  }

  return true;
}

bool kalinin_d_matrix_mult_hor_a_vert_b_mpi::ParallelMatrixMultiplicationTask::validation() {
  internal_order_test();

  if (_world.rank() == 0) {
    int rows_a = *reinterpret_cast<int*>(taskData->inputs[2]);
    int columns_a = *reinterpret_cast<int*>(taskData->inputs[3]);
    int columns_b = *reinterpret_cast<int*>(taskData->inputs[4]);

    bool is_big_inputs_count = taskData->inputs_count.size() > 3;
    bool is_empty_outputs = taskData->outputs_count.empty();
    bool has_zero_values = rows_a * columns_a * columns_b == 0;

    return (is_big_inputs_count && !is_empty_outputs && !has_zero_values);
  }
  return true;
}

bool kalinin_d_matrix_mult_hor_a_vert_b_mpi::ParallelMatrixMultiplicationTask::run() {
  internal_order_test();

  boost::mpi::broadcast(_world, _matrixA, 0);
  boost::mpi::broadcast(_world, _matrixB, 0);
  boost::mpi::broadcast(_world, _sizes, 0);
  boost::mpi::broadcast(_world, _displs, 0);

  int local_size = _sizes[_world.rank()];

  std::vector<int> local_indexes_a(local_size);
  std::vector<int> local_indexes_b(local_size);

  if (_world.rank() == 0) {
    boost::mpi::scatterv(_world, _indexesA.data(), _sizes, _displs, local_indexes_a.data(), local_size, 0);
    boost::mpi::scatterv(_world, _indexesB.data(), _sizes, _displs, local_indexes_b.data(), local_size, 0);
  } else {
    boost::mpi::scatterv(_world, local_indexes_a.data(), local_size, 0);
    boost::mpi::scatterv(_world, local_indexes_b.data(), local_size, 0);
  }

  std::vector<int> local_result(local_size, 0);

  for (size_t k = 0; k < local_indexes_a.size(); ++k) {
    int i = local_indexes_a[k];
    int j = local_indexes_b[k];

    auto itA = _matrixA.row_begin(i);
    auto itB = _matrixB.column_begin(j);

    while (itA != _matrixA.row_end(i) && itB != _matrixB.column_end(j)) {
      local_result[k] += (*itA) * (*itB);
      ++itA;
      ++itB;
    }
  }

  if (_world.rank() == 0) {
    boost::mpi::gatherv(_world, local_result.data(), local_result.size(), _result_vector.data(), _sizes, _displs, 0);
  } else {
    boost::mpi::gatherv(_world, local_result.data(), local_result.size(), 0);
  }

  return true;
}

bool kalinin_d_matrix_mult_hor_a_vert_b_mpi::ParallelMatrixMultiplicationTask::post_processing() {
  internal_order_test();

  if (_world.rank() == 0) {
    int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(_result_vector.begin(), _result_vector.end(), output_data);
  }

  return true;
}
