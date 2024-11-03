// Copyright 2024 Anikin Maksim
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

namespace anikin_m_summ_of_different_symbols_mpi {

std::string getRandomString(int sz);

class SumDifSymMPISequential : public ppc::core::Task {
 public:
  explicit SumDifSymMPISequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<char*> input;
  int res;
};

class SumDifSymMPIParallel : public ppc::core::Task {
 public:
  explicit SumDifSymMPIParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::string> local_input;
  std::vector<char*> input;
  int res;
  boost::mpi::communicator com;
};

}  // namespace anikin_m_summ_of_different_symbols_mpi