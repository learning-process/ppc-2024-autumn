#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <stdexcept>
#include <vector>
namespace khasanyanov_k_fox_algorithm {

template <typename T>
struct matrix {
  size_t rows, columns;
  std::vector<T> data;

  matrix() = default;
  matrix(size_t);
  matrix(size_t, size_t);
  matrix(size_t, size_t, const std::vector<T>&);

  auto begin() const { return data.begin(); }
  auto end() const { return data.end(); }
  [[nodiscard]] size_t size() const { return data.size(); };

  T& operator[](size_t);
  const T& operator[](size_t) const;
  matrix& operator+=(const matrix&);
  auto* operator*() { return data.data(); };

  friend bool operator==(const matrix<T>& lhs, const matrix<T>& rhs) { return lhs.data == rhs.data; }
};

template <typename T>
using BlockGrid = std::vector<matrix<T>>;

template <typename T>
matrix<T>::matrix(size_t rows) : matrix(rows, rows) {}

template <typename T>
matrix<T>::matrix(size_t rows, size_t columns) : rows(rows), columns(columns), data(std::vector<T>(rows * columns)) {}

template <typename T>
matrix<T>::matrix(size_t rows, size_t columns, const std::vector<T>& data) : rows(rows), columns(columns), data(data) {}

template <typename T>
T& matrix<T>::operator[](size_t id) {
  return data[id];
}

template <typename T>
const T& matrix<T>::operator[](size_t id) const {
  return data[id];
}

template <typename T>
matrix<T>& matrix<T>::operator+=(const matrix& rhs) {
  if (rows != rhs.rows || columns != rhs.columns) throw std::logic_error("Can't add matrix");
  for (size_t i = 0ull; i < size(); ++i) {
    data[i] += rhs[i];
  }
  return *this;
}

}  // namespace khasanyanov_k_fox_algorithm

#endif