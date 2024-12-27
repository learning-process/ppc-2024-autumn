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

namespace kozlova_e_lexic_order_mpi {

std::vector<int> LexicographicallyOrdered(const std::string& str1, const std::string& str2);

class StringComparatorSeq : public ppc::core::Task {
 public:
  explicit StringComparatorSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string str1{};
  std::string str2{};
  std::vector<int> res{};
};

class StringComparatorMPI : public ppc::core::Task {
 public:
  explicit StringComparatorMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::string> input_strings;
  std::vector<int> res;
  boost::mpi::communicator world;
};

}  // namespace kozlova_e_lexic_order_mpi