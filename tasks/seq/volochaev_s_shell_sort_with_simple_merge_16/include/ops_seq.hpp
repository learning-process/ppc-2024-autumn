#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace volochaev_s_shell_sort_with_simple_merge_16_seq {

class Lab3_16 : public ppc::core::Task {
 public:
  explicit Lab3_16(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  void merge(double* mas, int left, int right);

 private:
  int* mas;
  int size_{};
};

}  // namespace volochaev_s_shell_sort_with_simple_merge_16_seq
