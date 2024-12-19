#include <cassert>
#include <iostream>
#include <optional>
#include <ostream>
#include <ranges>
#include <utility>
#include <vector>

namespace krylov_m_crs_mmul {

namespace utils {
template <class InputIt, class Delimiter = char>
void iprint(std::ostream& os, InputIt first, InputIt last, Delimiter delimiter = ' ') {
  if (last == first) return;
  auto range = std::ranges::subrange{first, last};
  for (const auto& e : range | std::views::take(last - first - 1)) {
    os << e << delimiter;
  }
  os << range.back();
}
}  // namespace utils

template <class T>
struct Matrix {
  template <class TT>
  using Container = std::vector<TT>;

  size_t rows;
  size_t cols;
  Container<T> storage;

  bool check_integrity() const noexcept { return storage.size() == (rows * cols); }

  const T& at(size_t row, size_t col) const noexcept {
    const size_t idx = row * cols + col;
    assert(idx < storage.size());
    return storage[idx];
  }
  T& at(size_t row, size_t col) noexcept { return const_cast<T&>(std::as_const(*this).at(row, col)); }

  bool operator==(const Matrix& other) const noexcept {
    return rows == other.rows && cols == other.cols && storage == other.storage;
  }

  Matrix operator*(const Matrix& rhs) const {
    auto res = create(this->rows, rhs.cols);
    for (size_t i = 0; i < this->rows; i++) {
      for (size_t j = 0; j < rhs.cols; j++) {
        res.at(i, j) = 0;
        for (size_t k = 0; k < rhs.rows; k++) {
          res.at(i, j) += this->at(i, k) * rhs.at(k, j);
        }
      }
    }
    return res;
  }

  void read(const T* src) { storage.assign(src, src + rows * cols); }

  friend std::ostream& operator<<(std::ostream& os, const Matrix& m) {
    os << "M(" << m.rows << "," << m.cols << "): [";
    utils::iprint(os, m.storage.begin(), m.storage.end());
    os << ']';
    return os;
  }

  static Matrix create(size_t rows, size_t cols) { return {rows, cols, Container<T>(rows * cols)}; }
};

template <typename T>
struct CRSMatrix {
  std::vector<size_t> row_pointers;
  std::vector<size_t> col_indices;
  std::vector<T> data;
  size_t cols_;

  //

  CRSMatrix() = default;
  CRSMatrix(size_t rows, size_t cols) : row_pointers(rows + 1, 0), cols_(cols) {}
  CRSMatrix(std::vector<size_t> row_pointers_, std::vector<size_t> col_indices_, std::vector<T> data_, size_t cols)
      : row_pointers(std::move(row_pointers_)),
        col_indices(std::move(col_indices_)),
        data(std::move(data_)),
        cols_(cols) {}

  explicit CRSMatrix(const Matrix<T>& dense) : CRSMatrix(dense.rows, dense.cols) {
    size_t idx = 0;
    for (size_t i = 0; i < dense.rows; ++i) {
      size_t nz = 0;
      for (size_t j = 0; j < dense.cols; ++j) {
        const auto& el = dense.storage[idx++];
        if (el == 0) {
          continue;
        }
        ++nz;
        col_indices.push_back(j);
        data.push_back(el);
      }
      row_pointers[i + 1] = row_pointers[i] + nz;
    }
  }

  size_t rows() const { return row_pointers.size() - 1; }
  size_t cols() const { return cols_; }

  bool operator==(const CRSMatrix& other) const noexcept {
    return cols_ == other.cols_ && row_pointers == other.row_pointers && col_indices == other.col_indices &&
           data == other.data;
  }

  std::optional<const T&> at(size_t row, size_t col) const {
    for (size_t i = row_pointers[row]; i < row_pointers[row + 1]; ++i) {
      if (col_indices[i] == col) {
        return data[i];
      }
    }
    return std::nullopt;
  }

  CRSMatrix transpose() const {
    const auto [n, m] = std::make_pair(rows(), cols());

    CRSMatrix res{m + 1, n};
    res.col_indices.resize(col_indices.size(), 0);
    res.data.resize(data.size(), 0);

    for (size_t i = 0; i < data.size(); ++i) {
      ++res.row_pointers[col_indices[i] + 2];
    }

    for (size_t i = 2; i < res.row_pointers.size(); ++i) {
      res.row_pointers[i] += res.row_pointers[i - 1];
    }

    for (size_t i = 0; i < n; ++i) {
      for (size_t j = row_pointers[i]; j < row_pointers[i + 1]; ++j) {
        const auto translated_idx = res.row_pointers[col_indices[j] + 1]++;
        res.data[translated_idx] = data[j];
        res.col_indices[translated_idx] = i;
      }
    }
    res.row_pointers.pop_back();

    return res;
  }

  Matrix<T> densify() const {
    auto dense = Matrix<T>::create(rows(), cols());
    for (size_t row = 0; row < dense.rows; ++row) {
      for (size_t i = row_pointers[row]; i < row_pointers[row + 1]; ++i) {
        dense.at(row, col_indices[i]) = data[i];
      }
    }
    return dense;
  }

  friend std::ostream& operator<<(std::ostream& os, const CRSMatrix& m) {
    os << "CRS(" << m.rows() << "," << m.cols() << "): ";
    os << "R/[";
    utils::iprint(os, m.row_pointers.begin(), m.row_pointers.end());
    os << "], C/[";
    utils::iprint(os, m.col_indices.begin(), m.col_indices.end());
    os << "], D/[";
    utils::iprint(os, m.data.begin(), m.data.end());
    os << "]";
    return os;
  }

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & row_pointers;
    ar & col_indices;
    ar & data;
    ar & cols_;
  }
};

}  // namespace krylov_m_crs_mmul