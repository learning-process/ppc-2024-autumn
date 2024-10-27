#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vasenkov_a_char_freq_mpi {

std::vector<int> getRandomVector(int sz);

class CharFrequencySequential : public ppc::core::Task {
 public:
  explicit CharFrequencySequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<char> str_input_;
  char target_char_;
  int res{};
};

class CharFrequencyParallel : public ppc::core::Task {
 public:
  explicit CharFrequencyParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<char> str_input_;
  std::vector<char> local_input_;
  char target_char_;
  int res{};
  int local_res{};

  boost::mpi::communicator world;
};

}  // namespace vasenkov_a_char_freq_mpi