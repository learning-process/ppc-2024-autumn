#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace kozlova_e_radix_batcher_sort_seq {

class RadixSortSequential : public ppc::core::Task {
 public:
  explicit RadixSortSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int input_size{};
  std::vector<double> data;

  static void radixSort(std::vector<double>& a);
};

}  // namespace kozlova_e_radix_batcher_sort_seq