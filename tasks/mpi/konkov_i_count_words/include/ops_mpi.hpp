// ops_mpi.hpp
#pragma once

#include <boost/mpi/communicator.hpp>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace konkov_i_count_words_mpi {

class CountWordsTaskParallel : public ppc::core::Task {
 public:
  explicit CountWordsTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string input_;
  int word_count_{};
  boost::mpi::communicator world;
};

}  // namespace konkov_i_count_words_mpi
