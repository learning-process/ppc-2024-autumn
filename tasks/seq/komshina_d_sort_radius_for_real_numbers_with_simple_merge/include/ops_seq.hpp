#pragma once
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> input;
  std::vector<double> sort;
};

void SortDouble(std::vector<double>& data);
void CountingSort(double* inp, double* out, int byteNum, int size);
bool CompareArrays(double* mas, double* gMas, int size);
}  // namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq