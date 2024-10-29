// Filateva Elizaveta Number_of_sentences_per_line
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

namespace filateva_e_number_sentences_line_mpi {

std::string getRandomLine(int max_count);
int countSentences(std::string line);

class NumberSentencesLineSequential : public ppc::core::Task {
 public:
  explicit NumberSentencesLineSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string line;
  int sentence_count;
};

class NumberSentencesLineParallel : public ppc::core::Task {
 public:
  explicit NumberSentencesLineParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string line;
  std::string local_line;
  int sentence_count;
  boost::mpi::communicator world;
};

}  // namespace filateva_e_number_sentences_line_mpi