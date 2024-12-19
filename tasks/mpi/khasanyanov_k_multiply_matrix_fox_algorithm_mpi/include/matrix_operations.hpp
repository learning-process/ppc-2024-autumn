#ifndef _MATRIX_OPERATIONS_H_
#define _MATRIX_OPERATIONS_H_

#include <random>
#include <stdexcept>

#include "matrix.hpp"

namespace khasanyanov_k_fox_algorithm {

struct MatrixOperations {
  template <typename T>
  static matrix<T> multiply(const matrix<T>&, const matrix<T>&);
  template <typename T>
  inline static bool can_multiply(const matrix<T>&, const matrix<T>&);
  template <typename T>
  static matrix<T> generate_random_matrix(size_t rows = 3, size_t columns = 3, const T& left = static_cast<T>(-1000),
                                          const T& right = static_cast<T>(1000));
};

// not included 'right' border with integers, not included 'left' border always
template <typename T>
matrix<T> MatrixOperations::generate_random_matrix(size_t rows, size_t columns, const T& left, const T& right) {
  size_t size = rows * columns;
  std::vector<T> res(size);
  std::random_device dev;
  std::mt19937 gen(dev());
  double frac = (gen() % 100) / 100.0;
  for (size_t i = 0; i < size; i++) {
    res[i] = left + frac + static_cast<T>(gen() % static_cast<int>(right - left));
  }
  return {rows, columns, res};
}

template <typename T>
bool MatrixOperations::can_multiply(const matrix<T>& lhs, const matrix<T>& rhs) {
  return lhs.columns == rhs.rows;
}

template <typename T>
matrix<T> MatrixOperations::multiply(const matrix<T>& lhs, const matrix<T>& rhs) {
  if (!MatrixOperations::can_multiply(lhs, rhs)) {
    throw std::logic_error("can`t multiply matrix");
  }

  matrix<T> res(lhs.rows, rhs.columns);
  for (size_t i = 0; i < lhs.rows; ++i) {
    for (size_t j = 0; j < rhs.columns; ++j) {
      for (size_t k = 0; k < lhs.columns; ++k)
        res[i * rhs.columns + j] += lhs[i * lhs.columns + k] * rhs[k * rhs.columns + j];
    }
  }
  return res;
}

}  // namespace khasanyanov_k_fox_algorithm

#endif