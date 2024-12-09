#pragma once

#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace korneeva_e_num_of_orderly_violations_seq {

template <class iotype, class cntype>
class OrderlyViolationsCounter : public ppc::core::Task {
 public:
  explicit OrderlyViolationsCounter(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  cntype count_orderly_violations(const std::vector<iotype>& data);

 private:
  std::vector<iotype> input_;  // Input vector
  cntype result_;              // Number of violations
};

template <typename iotype, typename cntype>
bool OrderlyViolationsCounter<iotype, cntype>::pre_processing() {
  internal_order_test();

  int n = taskData->inputs_count[0];
  input_.resize(n);
  void* ptr_r = taskData->inputs[0];
  std::copy(static_cast<iotype*>(ptr_r), static_cast<iotype*>(ptr_r) + n, input_.begin());

  result_ = 0;
  return true;
}

template <typename iotype, typename cntype>
bool OrderlyViolationsCounter<iotype, cntype>::validation() {
  internal_order_test();
  return (taskData && taskData->inputs_count[0] > 0 && !taskData->inputs.empty() && taskData->outputs_count[0] == 1);
}

template <typename iotype, typename cntype>
bool OrderlyViolationsCounter<iotype, cntype>::run() {
  internal_order_test();
  result_ = count_orderly_violations(input_);
  return true;
}

template <typename iotype, typename cntype>
bool OrderlyViolationsCounter<iotype, cntype>::post_processing() {
  internal_order_test();
  if (!taskData || taskData->outputs.empty()) {
    return false;
  }
  *reinterpret_cast<cntype*>(taskData->outputs[0]) = result_;
  return true;
}

template <typename iotype, typename cntype>
cntype OrderlyViolationsCounter<iotype, cntype>::count_orderly_violations(const std::vector<iotype>& data) {
  cntype count = 0;
  for (size_t i = 1; i < data.size(); ++i) {
    if (data[i - 1] > data[i]) {
      ++count;
    }
  }
  return count;
}
}  // namespace korneeva_e_num_of_orderly_violations_seq