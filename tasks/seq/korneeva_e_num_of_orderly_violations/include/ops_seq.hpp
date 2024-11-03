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

template class korneeva_e_num_of_orderly_violations_seq::OrderlyViolationsCounter<int, int>;
template class korneeva_e_num_of_orderly_violations_seq::OrderlyViolationsCounter<double, int>;
}  // namespace korneeva_e_num_of_orderly_violations_seq