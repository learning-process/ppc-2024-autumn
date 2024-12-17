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

namespace makhov_m_num_of_diff_elements_in_two_str_mpi {

int countDiffElem(const std::string& str1_, const std::string& str2_);
std::string getShorterStr(std::string str1_, std::string str2_);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string str1{};
  std::string str2{};
  int res{};
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string str1{}, str1_local{}, str2{}, str2_local{};
  int sizeDiff{};
  int res{};
  boost::mpi::communicator world;
};

}  // namespace makhov_m_num_of_diff_elements_in_two_str_mpi