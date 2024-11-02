// Copyright 2023 Nesterov Alexander

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

namespace kazunin_n_count_freq_a_char_in_string_mpi {
class CharFreqCounterMPISequential : public ppc::core::Task {
 public:
  explicit CharFreqCounterMPISequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool run() override;
  bool validation() override;
  bool pre_processing() override;
  bool post_processing() override;

 private:
  size_t count_result_{0};
  char character_to_count_{};
  std::vector<char> input_string_;
};

class CharFreqCounterMPIParallel : public ppc::core::Task {
 public:
  explicit CharFreqCounterMPIParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool run() override;
  bool validation() override;
  bool pre_processing() override;
  bool post_processing() override;

 private:
  size_t total_count_{0};
  size_t local_count_{0};
  char character_to_count_{};
  std::vector<char> input_string_;
  std::vector<char> local_segment_;
  boost::mpi::communicator global;
};
}  // namespace kazunin_n_count_freq_a_char_in_string_mpi