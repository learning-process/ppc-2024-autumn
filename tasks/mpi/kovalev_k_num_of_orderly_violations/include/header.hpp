// Copyright 2023 Nesterov Alexander
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace kovalev_k_num_of_orderly_violations_mpi {
template <class T>
class NumOfOrderlyViolationsPar : public ppc::core::Task {
 private:
  std::vector<T> glob_v;
  std::vector<T> loc_v;
  size_t n = 0, l_res = 0, g_res = 0;  // counters initialisation
  int rank, size;
  boost::mpi::communicator world;

 public:
  explicit NumOfOrderlyViolationsPar(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(taskData_) {}
  bool count_num_of_orderly_violations_mpi();
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
};
}  // namespace kovalev_k_num_of_orderly_violations_mpi