#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace kazunin_n_quicksort_simple_merge_seq {

class QuicksortSimpleMergeSeq : public ppc::core::Task {
 public:
  explicit QuicksortSimpleMergeSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_vector;
};

}  // namespace kazunin_n_quicksort_simple_merge_seq
