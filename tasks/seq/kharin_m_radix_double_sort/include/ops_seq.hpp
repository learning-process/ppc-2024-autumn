#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace kharin_m_radix_double_sort {

// Класс для последовательной поразрядной сортировки double
class RadixSortSequential : public ppc::core::Task {
 public:
  explicit RadixSortSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> data;
  int n = 0;

  static void radix_sort_doubles(std::vector<double>& data);
  static void radix_sort_uint64(std::vector<uint64_t>& keys);
};

}  // namespace kharin_m_radix_double_sort