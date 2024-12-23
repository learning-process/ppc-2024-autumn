#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace volochaev_s_shell_sort_with_simple_merge_16_mpi {

class Lab3_16_seq : public ppc::core::Task {
 public:
  explicit Lab3_16_seq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> mas;
  int size_{};
};

class Lab3_16_mpi : public ppc::core::Task {
 public:
  explicit Lab3_16_mpi(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int size_{};

  std::vector<int> mas;
  std::vector<int> local_input;
  boost::mpi::communicator world;
};

}  // namespace volochaev_s_shell_sort_with_simple_merge_16_mpi
