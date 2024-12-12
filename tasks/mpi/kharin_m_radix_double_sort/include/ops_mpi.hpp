#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/serialization/vector.hpp>
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

// Класс для параллельной поразрядной сортировки double
class RadixSortParallel : public ppc::core::Task {
 public:
  explicit RadixSortParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> data;
  int n = 0;
  boost::mpi::communicator world;

  static void radix_sort_doubles(std::vector<double>& data);
  static void radix_sort_uint64(std::vector<uint64_t>& keys);
};

}  // namespace kharin_m_radix_double_sort