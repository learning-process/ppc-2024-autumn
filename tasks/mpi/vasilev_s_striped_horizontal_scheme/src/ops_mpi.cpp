#include "mpi/vasilev_s_striped_horizontal_scheme/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

void vasilev_s_striped_horizontal_scheme_mpi::calculate_distribution(int rows, int cols, int num_proc,
                                                                     std::vector<int>& sizes,
                                                                     std::vector<int>& displs) {
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

bool vasilev_s_striped_horizontal_scheme_mpi::StripedHorizontalSchemeParallelMPI::validation() {
  internal_order_test();
  if (world.rank() != 0) return false;

  bool valid_matrix = taskData->inputs[0] != nullptr && taskData->inputs_count[0] > 0;
  bool valid_vector = taskData->inputs[1] != nullptr && taskData->inputs_count[1] > 0;
  bool valid_dimensions = valid_matrix && valid_vector && taskData->inputs_count[0] % taskData->inputs_count[1] == 0;
  bool valid_result =
      valid_dimensions && taskData->outputs_count[0] == taskData->inputs_count[0] / taskData->inputs_count[1];

  if (valid_result) {
    std::vector<int> valid_sizes;
    std::vector<int> valid_displs;

    size_t valid_cols = taskData->inputs_count[1];
    size_t valid_rows = taskData->inputs_count[0] / taskData->inputs_count[1];
    size_t valid_num_proc = world.size();

    vasilev_s_striped_horizontal_scheme_mpi::calculate_distribution(valid_rows, valid_cols, valid_num_proc, valid_sizes,
                                                                    valid_displs);

    bool sizes_eq_num_proc = valid_sizes.size() == valid_num_proc;
    bool displs_eq_num_proc = valid_displs.size() == valid_num_proc;

    size_t i;
    for (i = 0; i < valid_num_proc; ++i) {
      if (i < valid_rows) {
        if (valid_sizes[i] == static_cast<int>(valid_cols)) break;
        if (valid_displs[i] == static_cast<int>(i * valid_cols)) break;
      } else {
        if (valid_sizes[i] == 0) break;
        if (valid_displs[i] == 0) break;
      }
    }
    bool flag = i == valid_num_proc - 1;

    return sizes_eq_num_proc && displs_eq_num_proc && flag;
  }
  return false;
}

bool vasilev_s_striped_horizontal_scheme_mpi::StripedHorizontalSchemeParallelMPI::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    int* matrix_data = reinterpret_cast<int*>(taskData->inputs[0]);
    int matrix_size = taskData->inputs_count[0];

    int* vector_data = reinterpret_cast<int*>(taskData->inputs[1]);
    int vector_size = taskData->inputs_count[1];

    input_matrix_.assign(matrix_data, matrix_data + matrix_size);
    input_vector_.assign(vector_data, vector_data + vector_size);

    num_cols_ = input_vector_.size();
    num_rows_ = input_matrix_.size() / num_cols_;

    int result_size = taskData->outputs_count[0];
    result_vector_.resize(result_size, 0);

    calculate_distribution(num_rows_, num_cols_, world.size(), distribution, displacement);
  }

  return true;
}

bool vasilev_s_striped_horizontal_scheme_mpi::StripedHorizontalSchemeParallelMPI::run() {
  internal_order_test();

  boost::mpi::broadcast(world, num_cols_, 0);
  boost::mpi::broadcast(world, input_vector_, 0);
  boost::mpi::broadcast(world, distribution, 0);

  int local_num_elements = distribution[world.rank()];
  int local_num_rows = local_num_elements / num_cols_;

  std::vector<int> local_matrix(local_num_elements);

  if (world.rank() == 0) {
    boost::mpi::scatterv(world, input_matrix_.data(), distribution, displacement, local_matrix.data(),
                         local_num_elements, 0);
  } else {
    boost::mpi::scatterv(world, local_matrix.data(), local_num_elements, 0);
  }

  std::vector<int> local_result(local_num_rows, 0);

  for (int i = 0; i < local_num_rows; ++i) {
    for (int j = 0; j < num_cols_; ++j) {
      local_result[i] += local_matrix[i * num_cols_ + j] * input_vector_[j];
    }
  }

  std::vector<int> gather_counts;
  std::vector<int> gather_displacements;

  if (world.rank() == 0) {
    gather_counts.resize(world.size());
    gather_displacements.resize(world.size());

    for (int i = 0; i < world.size(); ++i) {
      int num_elements = distribution[i] / num_cols_;
      gather_counts[i] = num_elements;
    }

    gather_displacements[0] = 0;
    for (int i = 1; i < world.size(); ++i) {
      gather_displacements[i] = gather_displacements[i - 1] + gather_counts[i - 1];
    }
  }

  if (world.rank() == 0) {
    boost::mpi::gatherv(world, local_result.data(), local_result.size(), result_vector_.data(), gather_counts,
                        gather_displacements, 0);
  } else {
    boost::mpi::gatherv(world, local_result.data(), local_result.size(), 0);
  }

  return true;
}

bool vasilev_s_striped_horizontal_scheme_mpi::StripedHorizontalSchemeParallelMPI::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(result_vector_.begin(), result_vector_.end(), output_data);
  }

  return true;
}

bool vasilev_s_striped_horizontal_scheme_mpi::StripedHorizontalSchemeSequentialMPI::validation() {
  internal_order_test();
  bool valid_matrix = taskData->inputs_count[0] > 0;
  bool valid_vector = taskData->inputs_count[1] > 0;
  bool valid_dimensions = valid_matrix && valid_vector && taskData->inputs_count[0] % taskData->inputs_count[1] == 0;
  bool valid_result =
      valid_dimensions && taskData->outputs_count[0] == taskData->inputs_count[0] / taskData->inputs_count[1];

  return valid_result;
}

bool vasilev_s_striped_horizontal_scheme_mpi::StripedHorizontalSchemeSequentialMPI::pre_processing() {
  internal_order_test();

  int* matrix_data = reinterpret_cast<int*>(taskData->inputs[0]);
  int matrix_size = taskData->inputs_count[0];

  int* vector_data = reinterpret_cast<int*>(taskData->inputs[1]);
  int vector_size = taskData->inputs_count[1];

  input_matrix_.assign(matrix_data, matrix_data + matrix_size);
  input_vector_.assign(vector_data, vector_data + vector_size);

  num_cols_ = input_vector_.size();
  num_rows_ = input_matrix_.size() / num_cols_;

  int result_size = taskData->outputs_count[0];
  result_vector_.resize(result_size, 0);

  return true;
}

bool vasilev_s_striped_horizontal_scheme_mpi::StripedHorizontalSchemeSequentialMPI::run() {
  internal_order_test();

  for (int i = 0; i < num_rows_; i++) {
    int sum = 0;
    for (int j = 0; j < num_cols_; j++) {
      sum += input_matrix_[i * num_cols_ + j] * input_vector_[j];
    }
    result_vector_[i] = sum;
  }

  return true;
}

bool vasilev_s_striped_horizontal_scheme_mpi::StripedHorizontalSchemeSequentialMPI::post_processing() {
  internal_order_test();

  int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(result_vector_.begin(), result_vector_.end(), output_data);

  return true;
}
