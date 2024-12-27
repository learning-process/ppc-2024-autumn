#pragma once

#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi {

class Matrix {
 public:
  Matrix() : rows_(0), cols_(0) {}

  Matrix(std::vector<int> matrix, size_t rows, size_t cols) : matrix_(std::move(matrix)), rows_(rows), cols_(cols) {}

  Matrix& operator=(Matrix&& other) noexcept {
    if (this != &other) {
      matrix_ = std::move(other.matrix_);
      rows_ = other.rows_;
      cols_ = other.cols_;
      other.rows_ = 0;
      other.cols_ = 0;
    }
    return *this;
  }

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & rows_;
    ar & cols_;
    ar & matrix_;
  }

  class RowIterator {
   public:
    RowIterator(const int* ptr) : ptr_(ptr) {}

    const int& operator*() const { return *ptr_; }
    RowIterator& operator++() {
      ++ptr_;
      return *this;
    }
    bool operator!=(const RowIterator& other) const { return ptr_ != other.ptr_; }

   private:
    const int* ptr_;
  };

  RowIterator row_begin(size_t row_index) const { return RowIterator(matrix_.data() + row_index * cols_); }

  RowIterator row_end(size_t row_index) const { return RowIterator(matrix_.data() + row_index * cols_ + cols_); }

  class ColumnIterator {
   public:
    ColumnIterator(const int* ptr, size_t step) : ptr_(ptr), step_(step) {}

    const int& operator*() const { return *ptr_; }
    ColumnIterator& operator++() {
      ptr_ += step_;
      return *this;
    }
    bool operator!=(const ColumnIterator& other) const { return ptr_ != other.ptr_; }

   private:
    const int* ptr_;
    size_t step_;
  };

  ColumnIterator col_begin(size_t col_index) const { return ColumnIterator(matrix_.data() + col_index, cols_); }

  ColumnIterator col_end(size_t col_index) const {
    return ColumnIterator(matrix_.data() + col_index + rows_ * cols_, cols_);
  }

  std::vector<int> matrix_;

 private:
  size_t rows_, cols_;
};

void get_indexes(int num_rows_a_, int num_rows_b_, std::vector<int>& indexesA, std::vector<int>& indexesB);
void calculate(int rows, int cols, int num_proc, std::vector<int>& sizes, std::vector<int>& displs);

class MatrixMultiplicationTaskSequential : public ppc::core::Task {
 public:
  explicit MatrixMultiplicationTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  Matrix matA;
  Matrix matB;

  int num_rows_a_;
  int num_cols_a_;
  int num_cols_b_;

  std::vector<int> result_vector_;
};

class MatrixMultiplicationTaskParallel : public ppc::core::Task {
 public:
  explicit MatrixMultiplicationTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int num_rows_a_;
  int num_cols_a_;
  int num_cols_b_;
  std::vector<int> indexesA_;
  std::vector<int> indexesB_;
  Matrix matA;
  Matrix matB;
  std::vector<int> sizes;
  std::vector<int> displs;

  std::vector<int> result_vector_;
  boost::mpi::communicator world;
};

}  // namespace shvedova_v_matrix_mult_horizontal_a_vertical_b_mpi
