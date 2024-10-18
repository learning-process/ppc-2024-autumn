#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <iterator>
#include <sstream>
#include <vector>
#include <memory>
#include <numeric>
#include <string>
#include <utility>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include "core/task/include/task.hpp"

namespace solovev_a_word_count_mpi {

std::string create_text(int quan_words);
int word_count(const std::string& input);

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
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  boost::mpi::communicator world;
  std::string input_;
  int res{};
  };

}  // namespace solovev_a_word_count_mpi