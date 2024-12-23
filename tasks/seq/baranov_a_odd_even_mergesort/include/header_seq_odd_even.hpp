#pragma once
#include <algorithm>
#include <cstring>
#include <random>
#include <stack>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
namespace baranov_a_qsort_odd_even_merge_seq {
template <class iotype>
class odd_even_mergesort_seq : public ppc::core::Task {
 public:
  explicit odd_even_mergesort_seq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(taskData_) {}
  std::vector<iotype> q_sort_stack(std::vector<iotype>& vec_);
  bool pre_processing() override;

  bool validation() override;

  bool run() override;

  bool post_processing() override;

 private:
  std::vector<iotype> input_;
  std::vector<iotype> output_;
};
}  // namespace baranov_a_qsort_odd_even_merge_seq
