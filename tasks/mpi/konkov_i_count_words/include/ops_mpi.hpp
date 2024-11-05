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

  static std::string generate_random_string(int num_words, int word_length);


 private:
  std::string input_;
  int word_count_{};
  boost::mpi::communicator world;
};

std::string generate_large_string(int num_words, int word_length);

}  // namespace konkov_i_count_words_mpi