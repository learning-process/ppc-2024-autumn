#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace muradov_m_count_alpha_chars_mpi {

class AlphaCharCountTaskSequential : public ppc::core::Task {
 public:
  explicit AlphaCharCountTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<char> input_str_;
  int alpha_count_ = 0;
};

class AlphaCharCountTaskParallel : public ppc::core::Task {
 public:
  explicit AlphaCharCountTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<char> input_str_;
  std::vector<char> local_input_;
  int local_alpha_count_ = 0;
  int total_alpha_count_ = 0;

  boost::mpi::communicator world;
};

}  // namespace muradov_m_count_alpha_chars_mpi
