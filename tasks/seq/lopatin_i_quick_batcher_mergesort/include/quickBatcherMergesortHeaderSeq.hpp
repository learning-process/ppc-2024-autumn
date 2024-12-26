#pragma once

#include <random>

#include "core/task/include/task.hpp"

namespace lopatin_i_quick_batcher_mergesort_seq {

void quicksort(std::vector<int>& arr, int low, int high);
int partition(std::vector<int>& arr, int low, int high);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> inputArray_;
  std::vector<int> resultArray_;

  int sizeArray;
};

}  // namespace lopatin_i_quick_batcher_mergesort_seq