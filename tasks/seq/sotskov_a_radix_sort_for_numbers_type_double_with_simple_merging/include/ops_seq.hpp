#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> input_data_;
  std::vector<double> sorted_data_;
};

void radixSortWithSignHandling(std::vector<double>& data);
void radixSort(std::vector<double>& data, int num_bits, int radix);

}  // namespace sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_seq
