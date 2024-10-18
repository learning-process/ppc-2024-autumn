#pragma once

#include <algorithm>
#include <boost/serialization/string.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstring>
#include <iterator>
#include <sstream>
#include <vector>

#include "core/task/include/task.hpp"

namespace lopatin_i_count_words_mpi {

int countWords(const std::string& str);
int divideWords(const std::vector<std::string>& words, int rank, int size);
std::string generateLongString(int n);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string input_;
  int word_count{};
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string input_;
  int word_count{};
  boost::mpi::communicator world;
};

}  // namespace lopatin_i_count_words_mpi
