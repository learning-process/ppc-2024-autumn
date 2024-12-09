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

namespace voroshilov_v_num_of_alphabetic_chars_mpi {

class AlphabetCharsTaskSequential : public ppc::core::Task {
 public:
  explicit AlphabetCharsTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<char> input_;
  int res_;
};

class AlphabetCharsTaskParallel : public ppc::core::Task {
 public:
  explicit AlphabetCharsTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<char> local_input_;
  int res_;
  boost::mpi::communicator world;
};

}  // namespace voroshilov_v_num_of_alphabetic_chars_mpi