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

namespace volochaev_s_count_characters_27_mpi {

std::vector<char> get_random_string(int sz);

class Lab1_27_seq: public ppc::core::Task {
 public:
  explicit Lab1_27_seq(std::shared_ptr<ppc::core::TaskData> taskData_, std::string ops_)
      : Task(std::move(taskData_)), ops(std::move(ops_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<char> input1_;
  std::vector<char> input2_;
  int res{};
  std::string ops;
};

class Lab1_27_mpi : public ppc::core::Task {
 public:
  explicit Lab1_27_mpi(std::shared_ptr<ppc::core::TaskData> taskData_, std::string ops_)
      : Task(std::move(taskData_)), ops(std::move(ops_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::pair<char,char>> input_, local_input_;
  int res{};
  std::string ops;
  boost::mpi::communicator world;
};

}  // namespace nesterov_a_test_task_mpi