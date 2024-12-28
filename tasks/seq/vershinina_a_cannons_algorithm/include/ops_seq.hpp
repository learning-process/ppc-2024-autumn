#pragma once

#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vershinina_a_cannons_algorithm {

template <class T>
struct TMatrix {
  size_t n;

  std::vector<T> data{};
  size_t hshift{};
  size_t vshift{};

  void set_horizontal_shift(size_t shift) { hshift = shift; }
  void set_vertical_shift(size_t shift) { vshift = shift; }

  const T& at(size_t row, size_t col) const noexcept { return data[row * n + col]; }
  T& at(size_t row, size_t col) noexcept { return const_cast<T&>(std::as_const(*this).at(row, col)); }

  const T& at_h(size_t row, size_t col) const noexcept {
    size_t actual_hshift = (hshift + row) % n;
    if (col < n - actual_hshift) {
      col += actual_hshift;
    } else {
      col = col - (n - actual_hshift);
    }
    return data[row * n + col];
  }
  T& at_h(size_t row, size_t col) noexcept { return const_cast<T&>(std::as_const(*this).at_h(row, col)); }

  const T& at_v(size_t row, size_t col) const noexcept {
    size_t actual_vshift = (vshift + col) % n;
    if (row < n - actual_vshift) {
      row += actual_vshift;
    } else {
      row = row - (n - actual_vshift);
    }
    return data[row * n + col];
  }
  T& at_v(size_t row, size_t col) noexcept { return const_cast<T&>(std::as_const(*this).at_v(row, col)); }

  bool operator==(const TMatrix& other) const noexcept { return n == other.n && data == other.data; }

  void read(const T* src) { data.assign(src, src + n * n); }

  static TMatrix create(size_t n, std::initializer_list<T> intl = {}) {
    TMatrix mat = {n, std::vector<T>(intl)};
    mat.data.resize(n * n);
    return mat;
  }
  TMatrix operator*(const TMatrix& rhs) const {
    auto res = create(n);
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < rhs.n; j++) {
        res.at(i, j) = 0;
        for (size_t k = 0; k < rhs.n; k++) {
          res.at(i, j) += at(i, k) * rhs.at(k, j);
        }
      }
    }
    return res;
  }
};

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  int n{};

 private:
  TMatrix<double> lhs_{};
  TMatrix<double> rhs_{};
  TMatrix<double> res_{};
  TMatrix<double> res_c{};
};
}  // namespace vershinina_a_cannons_algorithm