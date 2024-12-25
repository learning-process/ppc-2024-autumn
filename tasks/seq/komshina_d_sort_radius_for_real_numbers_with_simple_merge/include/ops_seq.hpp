#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
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
  void SortDoubleByBits(std::vector<double>& data_);
  static void BitwiseCountingSort(std::vector<uint64_t>& keys, int shift);

  std::vector<double> input;
};

}  // namespace somov_i_bitwise_sorting_batcher_merge_seq
