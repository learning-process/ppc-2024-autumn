#pragma once

#include <algorithm>
#include <cstdint>
#include <ostream>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace krylov_m_matmul_strip_ha_vb_seq {

template <class T>
struct TMatrix {
  size_t rows;
  size_t cols;

  std::vector<T> data{};

  bool check_integrity() const noexcept { return data.size() == rows * cols; }

  inline const T& at(size_t row, size_t col) const noexcept { return data[row * cols + col]; }
  inline T& at(size_t row, size_t col) noexcept { return const_cast<T&>(std::as_const(*this).at(row, col)); }

  bool operator==(const TMatrix& other) const noexcept {
    return rows == other.rows && cols == other.cols && data == other.data;
  }

  void read(const T* src) { data.assign(src, src + rows * cols); }

  friend std::ostream& operator<<(std::ostream& os, const TMatrix& m) {
    os << "M(" << m.rows << "," << m.cols << "): [";
    for (const auto& e : m.data) {
      os << e << ' ';
    }
    os << ']';
    return os;
  }

  static TMatrix create(size_t rows, size_t cols) { return {rows, cols, std::vector<T>(rows * cols)}; }
};

template <class T>
class TaskSequential : public ppc::core::Task {
 public:
  using Matrix = TMatrix<T>;

  explicit TaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool validation() override {
    internal_order_test();

    return taskData->inputs_count.size() == 4 && taskData->outputs.size() == 3 && taskData->outputs_count.size() == 3 &&
           // (lhs.cols == rhs.rows)
           (taskData->inputs_count[1] == taskData->inputs_count[2]) &&
           // lhs.rows > 0 && lhs.cols > 0 && rhs.rows > 0 [&& rhs.cols > 0] - true by definition
           (taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 && taskData->inputs_count[2] > 0) &&
           // out.storage.size() == lhs.rows * rhs.cols
           (taskData->outputs_count[0] == taskData->inputs_count[0] * taskData->inputs_count[3]) &&
           // out.rows, out.cols
           (taskData->outputs_count[1] == 1 && taskData->outputs_count[2] == 1);
  }

  bool pre_processing() override {
    internal_order_test();

    input_ = std::make_pair<Matrix, Matrix>({taskData->inputs_count[0], taskData->inputs_count[1]},
                                            {taskData->inputs_count[2], taskData->inputs_count[3]});
    input_.first.read(reinterpret_cast<T*>(taskData->inputs[0]));
    input_.second.read(reinterpret_cast<T*>(taskData->inputs[1]));

    res_.rows = input_.first.rows;
    res_.cols = input_.second.rows;
    res_.data.resize(res_.rows * res_.cols);

    return true;
  }

  bool run() override {
    internal_order_test();

    const auto& [lhs, rhs] = input_;

    for (size_t i = 0; i < lhs.rows; i++) {
      for (size_t j = 0; j < rhs.cols; j++) {
        res_.at(i, j) = 0;
        for (size_t k = 0; k < rhs.rows; k++) {
          res_.at(i, j) += lhs.at(i, k) * rhs.at(k, j);
        }
      }
    }

    return true;
  }

  bool post_processing() override {
    internal_order_test();

    *reinterpret_cast<size_t*>(taskData->outputs[1]) = res_.rows;
    *reinterpret_cast<size_t*>(taskData->outputs[2]) = res_.cols;
    std::copy(res_.data.begin(), res_.data.end(), reinterpret_cast<T*>(taskData->outputs[0]));

    return true;
  }

 private:
  std::pair<Matrix, Matrix> input_{};
  Matrix res_;
};

template <class T>
void fill_task_data(ppc::core::TaskData& data, const TMatrix<T>& lhs, const TMatrix<T>& rhs, TMatrix<T>& out) {
  out.data.resize(lhs.rows * rhs.cols);

  data.inputs.emplace_back(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(lhs.data.data())));
  data.inputs_count.emplace_back(lhs.rows);
  data.inputs_count.emplace_back(lhs.cols);
  //
  data.inputs.emplace_back(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(rhs.data.data())));
  data.inputs_count.emplace_back(rhs.rows);
  data.inputs_count.emplace_back(rhs.cols);

  data.outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data.data()));
  data.outputs_count.emplace_back(out.data.size());
  //
  data.outputs.emplace_back(reinterpret_cast<uint8_t*>(&out.rows));
  data.outputs_count.emplace_back(1);
  //
  data.outputs.emplace_back(reinterpret_cast<uint8_t*>(&out.cols));
  data.outputs_count.emplace_back(1);
}

}  // namespace krylov_m_matmul_strip_ha_vb_seq
