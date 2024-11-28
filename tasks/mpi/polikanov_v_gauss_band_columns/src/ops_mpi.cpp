#include "mpi/polikanov_v_gauss_band_columns/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <cstddef>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

bool polikanov_v_gauss_band_columns_mpi::GaussBandColumnsParallelMPI::validation() {
  internal_order_test();
  bool result;
  if (world.rank() == 0) {
    size_t val_n = *reinterpret_cast<size_t*>(taskData->inputs[1]);
    size_t val_mat_size = taskData->inputs_count[0];

    result = val_n >= 2 && val_mat_size == (val_n * (val_n + 1));
  }
  boost::mpi::broadcast(world, result, 0);
  return result;
}

bool polikanov_v_gauss_band_columns_mpi::GaussBandColumnsParallelMPI::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* matrix_data = reinterpret_cast<double*>(taskData->inputs[0]);
    int matrix_size = taskData->inputs_count[0];
    n = *reinterpret_cast<int*>(taskData->inputs[1]);

    std::vector<double> matrix(n, n + 1);
    matrix.assign(matrix_data, matrix_data + matrix_size);

    mat = Matrix(matrix, n);
    answers.resize(n * (n + 1));
  }

  return true;
}

bool polikanov_v_gauss_band_columns_mpi::GaussBandColumnsParallelMPI::run() {
  internal_order_test();

  boost::mpi::broadcast(world, n, 0);

  size_t rank = world.rank();

  for (size_t k = 0; k < n - 1; k++) {
    if (rank == 0) {
      iter_mat = mat.submatrix(k, k);

      iter_mat.calc_sizes_displs(world.size(), counts, displs);

      try {
        factors = iter_mat.calculate_elimination_factors();
      } catch (const std::logic_error& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        break;
      }
    }

    size_t iter_mat_rows;
    if (rank == 0) {
      iter_mat_rows = iter_mat.get_rows();
    }
    boost::mpi::broadcast(world, iter_mat_rows, 0);

    boost::mpi::broadcast(world, counts, 0);
    boost::mpi::broadcast(world, factors, 0);

    std::vector<double> local_data(counts[rank]);
    std::vector<double> data_vector;

    if (rank == 0) {
      data_vector = iter_mat.to_vector();
      boost::mpi::scatterv(world, data_vector.data(), counts, displs, local_data.data(), local_data.size(), 0);
    } else {
      boost::mpi::scatterv(world, local_data.data(), local_data.size(), 0);
    }

    size_t local_size = local_data.size();
    size_t num_columns = local_size / iter_mat_rows;

    for (size_t col = 0; col < num_columns; ++col) {
      std::vector<double> column(iter_mat_rows);

      for (size_t i = 0; i < iter_mat_rows; ++i) {
        column[i] = local_data[col * iter_mat_rows + i];
      }

      double pivot = column[0];
      for (size_t i = 1; i < column.size(); ++i) {
        column[i] -= factors[i - 1] * pivot;
      }

      for (size_t i = 0; i < iter_mat_rows; ++i) {
        local_data[col * iter_mat_rows + i] = column[i];
      }
    }

    std::vector<double> result_vector;

    if (rank == 0) {
      result_vector.resize(data_vector.size());
      boost::mpi::gatherv(world, local_data.data(), local_data.size(), result_vector.data(), counts, displs, 0);
    } else {
      boost::mpi::gatherv(world, local_data.data(), local_data.size(), 0);
    }

    if (rank == 0) {
      iter_mat.from_vector(result_vector);
      mat.at(k, k) = 1;
    }
  }

  if (rank == 0) {
    answers = mat.backward_substitution();
  }

  return true;
}

bool polikanov_v_gauss_band_columns_mpi::GaussBandColumnsParallelMPI::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* output_data = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(answers.begin(), answers.end(), output_data);
  }

  return true;
}

bool polikanov_v_gauss_band_columns_mpi::GaussBandColumnsSequentialMPI::validation() {
  internal_order_test();

  size_t val_n = *reinterpret_cast<size_t*>(taskData->inputs[1]);
  size_t val_mat_size = taskData->inputs_count[0];

  return val_n >= 2 && val_mat_size == (val_n * (val_n + 1));
}

bool polikanov_v_gauss_band_columns_mpi::GaussBandColumnsSequentialMPI::pre_processing() {
  internal_order_test();

  auto* matrix_data = reinterpret_cast<double*>(taskData->inputs[0]);
  int matrix_size = taskData->inputs_count[0];
  n = *reinterpret_cast<size_t*>(taskData->inputs[1]);

  std::vector<double> matrix(n, n + 1);
  matrix.assign(matrix_data, matrix_data + matrix_size);

  mat = Matrix(matrix, n);
  answers.resize(n * (n + 1));

  return true;
}

bool polikanov_v_gauss_band_columns_mpi::GaussBandColumnsSequentialMPI::run() {
  internal_order_test();

  size_t n = mat.get_rows();

  for (size_t k = 0; k < n - 1; ++k) {
    Matrix iter_mat = mat.submatrix(k, k);

    std::vector<double> factors;
    try {
      factors = iter_mat.calculate_elimination_factors();
    } catch (const std::logic_error& e) {
      std::cerr << "\nError: " << e.what() << std::endl;
      return false;
    }

    for (size_t i = 1; i < iter_mat.get_rows(); ++i) {
      double factor = factors[i - 1];
      for (size_t j = 0; j < iter_mat.get_cols(); ++j) {
        iter_mat.at(i, j) -= factor * iter_mat.at(0, j);
      }
      iter_mat.at(i, 0) = 0.0;
    }
  }

  try {
    answers = mat.backward_substitution();
  } catch (const std::exception& e) {
    std::cerr << "\nError: " << e.what() << std::endl;
    return false;
  }

  return true;
}

bool polikanov_v_gauss_band_columns_mpi::GaussBandColumnsSequentialMPI::post_processing() {
  internal_order_test();

  auto* output_data = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(answers.begin(), answers.end(), output_data);

  return true;
}
