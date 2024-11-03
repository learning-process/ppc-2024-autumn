// Copyright 2024 Chastov Vyacheslav
#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace chastov_v_count_words_in_line_mpi {

std::vector<char> createString(int n);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<char> input_;
  int wordsFound{};
  int spacesFound{};
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<char> input_, local_input_;
  int localSpaceFound{};
  int wordsFound{};
  int spacesFound{};
  boost::mpi::communicator world;
};

}  // namespace chastov_v_count_words_in_line_mpi