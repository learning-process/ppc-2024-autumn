#pragma once

#include <cmath>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace korablev_v_qucik_sort_simple_merge_seq {

class QuickSortSimpleMergeSequential : public ppc::core::Task {
 public:
  explicit QuickSortSimpleMergeSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> input_;
  std::vector<double> output_;

  static std::vector<double> merge(const std::vector<double>& left, const std::vector<double>& right);
  std::vector<double> quick_sort_with_merge(const std::vector<double>& arr);
};

}  // namespace korablev_v_qucik_sort_simple_merge_seq