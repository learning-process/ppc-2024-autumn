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

namespace strakhov_a_str_char_freq_mpi {
class StringCharactersFrequencySequentional : public ppc::core::Task {
 public:
  explicit StringCharactersFrequencySequentional(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;


  bool run() override;
  bool post_processing() override;

 private:
  std::vector<char> input_;
  char target_;
  int res{};
};

class StringCharactersFrequencyParallel : public ppc::core::Task {
 public:
  explicit StringCharactersFrequencyParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<char> input_;
  std::vector<char> local_input_;
  char target_;
  int res{};
  int local_res{};
  boost::mpi::communicator world;
};

}  // namespace strakhov_a_str_char_freq_mpi