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

namespace kalinin_d_matrix_mult_hor_a_vert_b_mpi {

class Matrix {
 public:
  Matrix() : _rows(0), _columns(0) {}

  Matrix(std::vector<int> matrix_data, size_t rows, size_t columns)
      : _matrixData(std::move(matrix_data)), _rows(rows), _columns(columns) {}

  Matrix& operator=(Matrix&& other) noexcept {
    if (this != &other) {
      _matrixData = std::move(other._matrixData);
      _rows = other._rows;
      _columns = other._columns;
      other._rows = other._columns = 0;
    }
    return *this;
  }

  template <class Archive>
  void serialize(Archive& arhive, const unsigned int version) {
    arhive & _rows & _columns & _matrixData;
  }

  class RowIterator {
   public:
    RowIterator(const int* ptr) : _ptr(ptr) {}

    const int& operator*() const { return *_ptr; }
    RowIterator& operator++() {
      ++_ptr;
      return *this;
    }
    bool operator!=(const RowIterator& other) const { return _ptr != other._ptr; }

   private:
    const int* _ptr;
  };

  class ColumnIterator {
   public:
    ColumnIterator(const int* ptr, size_t step) : _ptr(ptr), _step(step) {}

    const int& operator*() const { return *_ptr; }
    ColumnIterator& operator++() {
      _ptr += _step;
      return *this;
    }
    bool operator!=(const ColumnIterator& other) const { return _ptr != other._ptr; }

   private:
    const int* _ptr;
    size_t _step;
  };

  RowIterator row_begin(size_t row_index) const { return RowIterator(_matrixData.data() + row_index * _columns); }

  RowIterator row_end(size_t row_index) const { return RowIterator(_matrixData.data() + (row_index + 1) * _columns); }

  ColumnIterator column_begin(size_t col_index) const {
    return ColumnIterator(_matrixData.data() + col_index, _columns);
  }

  ColumnIterator column_end(size_t col_index) const {
    return ColumnIterator(_matrixData.data() + col_index + _rows * _columns, _columns);
  }

  std::vector<int> _matrixData;

 private:
  size_t _rows, _columns;
};

void compute_indexes(int num_rows_a, int num_rows_b, std::vector<int>& indexesA, std::vector<int>& indexesB);
void calculate(int rows, int columns, int num_proc, std::vector<int>& _sizes, std::vector<int>& _displs);

class SequentialMatrixMultiplicationTask : public ppc::core::Task {
 public:
  explicit SequentialMatrixMultiplicationTask(std::shared_ptr<ppc::core::TaskData> taskData)
      : Task(std::move(taskData)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  Matrix _matrixA;
  Matrix _matrixB;

  int _rows_a;
  int _columns_a;
  int _columns_b;

  std::vector<int> _result_vector;
};

class ParallelMatrixMultiplicationTask : public ppc::core::Task {
 public:
  explicit ParallelMatrixMultiplicationTask(std::shared_ptr<ppc::core::TaskData> taskData)
      : Task(std::move(taskData)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int _rows_a;
  int _columns_a;
  int _columns_b;
  std::vector<int> _indexesA;
  std::vector<int> _indexesB;
  Matrix _matrixA;
  Matrix _matrixB;
  std::vector<int> _sizes;
  std::vector<int> _displs;

  std::vector<int> _result_vector;
  boost::mpi::communicator _world;
};

}  // namespace kalinin_d_matrix_mult_hor_a_vert_b_mpi
