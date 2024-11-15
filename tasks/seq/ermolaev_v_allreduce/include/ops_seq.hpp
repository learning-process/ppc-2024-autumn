// Copyright 2023 Nesterov Alexander
#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#include "core/task/include/task.hpp"

namespace ermolaev_v_allreduce_seq {

template <typename _T, typename _S = uint32_t>
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<_T>> input_, res_;
};

}  // namespace ermolaev_v_allreduce_seq

// TestTaskSequential
template <typename _T, typename _S>
bool ermolaev_v_allreduce_seq::TestTaskSequential<_T, _S>::pre_processing() {
  internal_order_test();

  input_.resize(taskData->inputs_count[0], std::vector<_T>(taskData->inputs_count[1]));

  auto* ptr = reinterpret_cast<std::shared_ptr<_T[]>*>(taskData->inputs[0]);
  for (_S i = 0; i < input_.size(); i++) {
    for (_S j = 0; j < input_[i].size(); j++) {
      input_[i][j] = ptr[i][j];
    }
  }

  res_.resize(taskData->inputs_count[0], std::vector<_T>(taskData->inputs_count[1]));

  return true;
}

template <typename _T, typename _S>
bool ermolaev_v_allreduce_seq::TestTaskSequential<_T, _S>::validation() {
  internal_order_test();

  return (taskData->inputs_count == taskData->outputs_count && !taskData->inputs.empty() && !taskData->outputs.empty());
}

template <typename _T, typename _S>
bool ermolaev_v_allreduce_seq::TestTaskSequential<_T, _S>::run() {
  internal_order_test();

  for (_S i = 0; i < input_.size(); i++) {
    const auto [min, max] = std::minmax_element(input_[i].begin(), input_[i].end());
    for (_S j = 0; j < input_[i].size(); j++) {
      if ((*max - *min) <= std::numeric_limits<_T>::epsilon())
        res_[i][j] = 0;
      else
        res_[i][j] = (input_[i][j] - *min) / (*max - *min);
    }
  }

  return true;
}

template <typename _T, typename _S>
bool ermolaev_v_allreduce_seq::TestTaskSequential<_T, _S>::post_processing() {
  internal_order_test();

  auto* ptr = reinterpret_cast<std::shared_ptr<_T[]>*>(taskData->outputs[0]);
  for (_S i = 0; i < res_.size(); i++) {
    std::copy(res_[i].begin(), res_[i].end(), ptr[i].get());
  }

  return true;
}
// TestTaskSequential
