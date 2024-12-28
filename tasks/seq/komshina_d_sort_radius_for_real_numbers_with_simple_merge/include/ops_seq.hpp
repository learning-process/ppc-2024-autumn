#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq {

void convert_doubles_to_uint64(const std::vector<double>& data_, std::vector<uint64_t>& keys);
void radix_sort_uint64(std::vector<uint64_t>& keys);
void convert_uint64_to_doubles(const std::vector<uint64_t>& keys, std::vector<double>& data_);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> data;
};
}  // namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq