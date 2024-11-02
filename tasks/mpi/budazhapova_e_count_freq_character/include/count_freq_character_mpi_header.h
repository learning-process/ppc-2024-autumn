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

namespace budazhapova_e_count_freq_character_mpi{

  std::string getRandomString(int length);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  
 private:
  std::string input_;
  int res{};
  char symb{},
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string input_, local_input_;
  int res{};
  char symb{}, local_symb{};
  int local_res{};
  boost::mpi::communicator world;
};

} 