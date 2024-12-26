#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#include "./matrix.hpp"
#include "core/task/include/task.hpp"

namespace krylov_m_crs_mmul {

template <typename T>
class TaskCommon : public ppc::core::Task {
 public:
  explicit TaskCommon(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool validation() override {
    internal_order_test();

    return taskData->inputs.size() == 2 && taskData->inputs_count.size() == 4 && taskData->outputs.size() == 1 &&
           // (lhs.cols == rhs.rows)
           (taskData->inputs_count[1] == taskData->inputs_count[2]) &&
           // lhs.rows > 0 && lhs.cols > 0 && rhs.rows > 0 [&& rhs.cols > 0] - true by definition
           (taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 && taskData->inputs_count[2] > 0);
  }

  bool pre_processing() override {
    internal_order_test();

    input = {*reinterpret_cast<CRSMatrix<T>*>(taskData->inputs[0]),
             reinterpret_cast<CRSMatrix<T>*>(taskData->inputs[1])->transpose()};

    return true;
  }

  bool post_processing() override {
    internal_order_test();

    *reinterpret_cast<CRSMatrix<T>*>(taskData->outputs[0]) = res;

    return true;
  }

 protected:
  std::pair<CRSMatrix<T>, CRSMatrix<T>> input;
  CRSMatrix<T> res;
};

template <typename T>
class TaskSequential : public TaskCommon<T> {
 public:
  explicit TaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : TaskSequential::TaskCommon(std::move(taskData_)) {}

  bool run() override {
    this->internal_order_test();

    const auto& [lhs, rhs] = this->input;

    const auto [rows, cols] = std::make_pair(lhs.rows(), rhs.rows());  // rhs was transposed
    this->res = decltype(this->res)(rows, cols);

    for (size_t row = 0; row < rows; row++) {
      for (size_t col = 0; col < cols; col++) {
        auto [il, ir] = std::make_pair(lhs.row_pointers[row], rhs.row_pointers[col]);
        T sum{};
        while (il < lhs.row_pointers[row + 1] && ir < rhs.row_pointers[col + 1]) {
          if (lhs.col_indices[il] < rhs.col_indices[ir]) {
            il++;
          } else if (lhs.col_indices[il] > rhs.col_indices[ir]) {
            ir++;
          } else {  // ==
            sum += lhs.data[il] * rhs.data[ir];
            il++;
            ir++;
          }
        }
        if (sum != 0) {
          this->res.data.push_back(sum);
          this->res.col_indices.push_back(col);
        }
      }
      this->res.row_pointers[row + 1] = this->res.data.size();
    }

    return true;
  }
};

template <class T>
void fill_task_data(ppc::core::TaskData& data, const CRSMatrix<T>& lhs, const CRSMatrix<T>& rhs, CRSMatrix<T>& out) {
  data.inputs.emplace_back(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(&lhs)));
  data.inputs_count.emplace_back(lhs.rows());
  data.inputs_count.emplace_back(lhs.cols());
  //
  data.inputs.emplace_back(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(&rhs)));
  data.inputs_count.emplace_back(rhs.rows());
  data.inputs_count.emplace_back(rhs.cols());

  data.outputs.emplace_back(reinterpret_cast<uint8_t*>(&out));
}

}  // namespace krylov_m_crs_mmul