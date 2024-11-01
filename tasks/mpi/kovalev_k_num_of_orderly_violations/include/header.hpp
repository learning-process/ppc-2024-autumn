// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>
#include <cstdlib>
#include <mpi.h>
#include <cstring>
#include "core/task/include/task.hpp"

namespace kovalev_k_num_of_orderly_violations_mpi {

template <class T>
class NumOfOrderlyViolationsPar : public ppc::core::Task {
 private:
  std::vector<T> v;
  size_t n, l_res, g_res;
  int rank, size;

 public:
  explicit NumOfOrderlyViolationsPar(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(taskData_), n(taskData_->inputs_count[0]), l_res(0), g_res(0) {}

  bool count_num_of_orderly_violations_mpi();

  bool pre_processing() override;

  bool validation() override;

  bool run() override;

  bool post_processing() override;
};
}  // namespace kovalev_k_num_of_orderly_violations_mpi