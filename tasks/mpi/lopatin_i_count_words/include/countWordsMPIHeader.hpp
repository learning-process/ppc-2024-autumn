#pragma once

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace lopatin_i_count_words_mpi {

std::vector<char> generateLongString(int n);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<char> input_;
  int wordCount{};
  int spaceCount{};
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<char> input_;
  std::vector<char> localInput_;
  int wordCount{};
  int spaceCount{};
  int localSpaceCount{};
  int chunkSize{};
  boost::mpi::communicator world;
};

}  // namespace lopatin_i_count_words_mpi