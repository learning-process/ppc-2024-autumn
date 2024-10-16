// Copyright 2024 Kabalova Valeria
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

namespace kabalova_v_count_symbols_mpi {

int getRandomNumber(int left, int right);
std::string getRandomString();
int countSymbols(std::string& str);

class Task1Seq : public ppc::core::Task {
 public:
  explicit Task1Seq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string input_{};
  int result{};
};

class Task1Mpi : public ppc::core::Task {
 public:
  explicit Task1Mpi(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string input_{}, local_input_{};
  int result{};
  boost::mpi::communicator world;
};



}  // namespace kabalova_v_count_symbols_seq