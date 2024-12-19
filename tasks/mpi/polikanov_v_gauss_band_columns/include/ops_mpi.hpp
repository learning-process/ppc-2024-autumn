#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/vector.hpp>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace polikanov_v_gauss_band_columns_mpi {

class Matrix {
 private:
  size_t rows, cols;
  size_t row_offset, col_offset;
  size_t parent_rows, parent_cols;
  std::shared_ptr<std::vector<double>> data;

  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & rows;
    ar & cols;
    ar & row_offset;
    ar & col_offset;
    ar & parent_rows;
    ar & parent_cols;
    ar & data;
  }

 public:
  void swap_rows(size_t i1, size_t i2) {
    if (i1 >= rows || i2 >= rows) {
      throw std::out_of_range("Row index out of bounds");
    }

    size_t global_i1 = row_offset + i1;
    size_t global_i2 = row_offset + i2;

    for (size_t j = 0; j < parent_cols; ++j) {
      size_t index1 = j * parent_rows + global_i1;
      size_t index2 = j * parent_rows + global_i2;
      std::swap((*data)[index1], (*data)[index2]);
    }
  }

  Matrix()
      : rows(0),
        cols(0),
        row_offset(0),
        col_offset(0),
        parent_rows(0),
        parent_cols(0),
        data(std::make_shared<std::vector<double>>()) {}

  Matrix(const std::vector<double>& input_matrix, size_t n)
      : rows(n),
        cols(n + 1),
        row_offset(0),
        col_offset(0),
        parent_rows(rows),
        parent_cols(cols),
        data(std::make_shared<std::vector<double>>(rows * cols)) {
    for (size_t j = 0; j < cols; ++j) {
      for (size_t i = 0; i < rows; ++i) {
        size_t index = j * parent_rows + i;
        (*data)[index] = input_matrix[i * cols + j];
      }
    }
  }

  Matrix(Matrix&& other) noexcept
      : rows(other.rows),
        cols(other.cols),
        row_offset(other.row_offset),
        col_offset(other.col_offset),
        parent_rows(other.parent_rows),
        parent_cols(other.parent_cols),
        data(std::move(other.data)) {
    other.rows = 0;
    other.cols = 0;
    other.row_offset = 0;
    other.col_offset = 0;
    other.parent_rows = 0;
    other.parent_cols = 0;
  }

  Matrix(const Matrix& other) = default;

  Matrix& operator=(const Matrix& other) {
    if (this != &other) {
      rows = other.rows;
      cols = other.cols;
      row_offset = other.row_offset;
      col_offset = other.col_offset;
      parent_rows = other.parent_rows;
      parent_cols = other.parent_cols;
      data = other.data;
    }
    return *this;
  }

  Matrix& operator=(Matrix&& other) noexcept {
    if (this != &other) {
      rows = other.rows;
      cols = other.cols;
      row_offset = other.row_offset;
      col_offset = other.col_offset;
      parent_rows = other.parent_rows;
      parent_cols = other.parent_cols;
      data = std::move(other.data);

      other.rows = 0;
      other.cols = 0;
      other.row_offset = 0;
      other.col_offset = 0;
      other.parent_rows = 0;
      other.parent_cols = 0;
    }
    return *this;
  }

  Matrix(size_t rows, size_t cols)
      : rows(rows),
        cols(cols),
        row_offset(0),
        col_offset(0),
        parent_rows(rows),
        parent_cols(cols),
        data(std::make_shared<std::vector<double>>(rows * cols)) {}

  double& at(size_t i, size_t j) {
    if (i >= rows || j >= cols) {
      throw std::out_of_range("Index out of bounds");
    }
    size_t global_i = row_offset + i;
    size_t global_j = col_offset + j;
    size_t index = global_j * parent_rows + global_i;
    return (*data)[index];
  }

  const double& at(size_t i, size_t j) const {
    if (i >= rows || j >= cols) {
      throw std::out_of_range("Index out of bounds");
    }
    size_t global_i = row_offset + i;
    size_t global_j = col_offset + j;
    size_t index = global_j * parent_rows + global_i;
    return (*data)[index];
  }

  void fill(const std::function<double(size_t, size_t)>& func) {
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        at(i, j) = func(i, j);
      }
    }
  }

  Matrix submatrix(size_t start_row, size_t start_col) {
    if (start_row >= rows || start_col >= cols) {
      throw std::out_of_range("Submatrix starting indices out of bounds");
    }
    auto m = *this;
    m.rows -= start_row;
    m.cols -= start_col;
    m.row_offset += start_row;
    m.col_offset += start_col;
    return m;
  }

  std::vector<double> to_vector() const {
    std::vector<double> result(rows * cols);
    size_t idx = 0;

    for (size_t j = 0; j < cols; ++j) {
      for (size_t i = 0; i < rows; ++i) {
        result[idx++] = at(i, j);
      }
    }

    return result;
  }

  void from_vector(const std::vector<double>& vec) {
    if (vec.size() != rows * cols) {
      throw std::invalid_argument("Размер вектора не соответствует размеру подматрицы.");
    }
    size_t idx = 0;

    for (size_t j = 0; j < cols; ++j) {
      for (size_t i = 0; i < rows; ++i) {
        at(i, j) = vec[idx++];
      }
    }
  }

  void calc_sizes_displs(size_t num_procs, std::vector<int>& counts, std::vector<int>& displs) const {
    counts.resize(num_procs);
    displs.resize(num_procs);

    size_t total_cols = cols;
    size_t total_rows = rows;

    size_t cols_per_proc = total_cols / num_procs;
    size_t remaining_cols = total_cols % num_procs;

    for (size_t proc = 0; proc < num_procs; ++proc) {
      size_t cols_for_proc = cols_per_proc + (proc < remaining_cols ? 1 : 0);
      counts[proc] = static_cast<int>(cols_for_proc * total_rows);

      size_t col_start = proc * cols_per_proc + std::min(proc, remaining_cols);
      size_t disp = col_start * total_rows;

      displs[proc] = static_cast<int>(disp);
    }
  }

  std::vector<double> calculate_elimination_factors() {
    if (rows < 2) {
      throw std::logic_error("Matrix must have at least two rows to perform elimination.");
    }

    double pivot = at(0, 0);

    if (pivot == 0) {
      size_t non_zero_row = rows;
      for (size_t i = 1; i < rows; ++i) {
        if (at(i, 0) != 0) {
          non_zero_row = i;
          break;
        }
      }

      if (non_zero_row == rows) {
        throw std::logic_error("All elements in the pivot column are zero. Cannot perform elimination.");
      }

      swap_rows(0, non_zero_row);
      pivot = at(0, 0);
    }

    for (size_t i = 0; i < cols; i++) {
      at(0, i) /= pivot;
    }

    std::vector<double> factors(rows - 1);

    for (size_t i = 1; i < rows; ++i) {
      factors[i - 1] = at(i, 0);
    }

    return factors;
  }

  std::vector<double> backward_substitution() const {
    size_t n = get_rows();
    size_t m = get_cols();

    if (m != n + 1) {
      throw std::invalid_argument("Матрица должна быть расширенной с n+1 столбцами.");
    }

    std::vector<double> x(n);

    for (size_t i = n; i-- > 0;) {
      double sum = 0.0;

      for (size_t j = i + 1; j < n; ++j) {
        sum += at(i, j) * x[j];
      }

      double diag = at(i, i);
      if (diag == 0.0) {
        throw std::runtime_error("Обнаружен нулевой диагональный элемент во время обратного хода.");
      }

      double bi = at(i, n);

      x[i] = (bi - sum) / diag;
    }

    return x;
  }

  double* col_data() { return data->data(); }

  const double* col_data() const { return data->data(); }

  size_t get_rows() const { return rows; }

  size_t get_cols() const { return cols; }

  void set_rows(size_t value) { rows = value; }

  void set_cols(size_t value) { cols = value; }

  size_t get_size() const { return data->size(); }
};

class GaussBandColumnsParallelMPI : public ppc::core::Task {
 public:
  explicit GaussBandColumnsParallelMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  Matrix mat;
  size_t n;
  Matrix iter_mat;
  std::vector<int> counts;
  std::vector<int> displs;
  std::vector<double> factors;
  std::vector<double> answers;
  boost::mpi::communicator world;
};

class GaussBandColumnsSequentialMPI : public ppc::core::Task {
 public:
  explicit GaussBandColumnsSequentialMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  Matrix mat;
  size_t n;
  std::vector<double> answers;
};

}  // namespace polikanov_v_gauss_band_columns_mpi
