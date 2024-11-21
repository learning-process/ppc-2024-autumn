#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include "core/task/include/task.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace rysev_m_count_of_sent_mpi {

class CountOfSentSeq : public ppc::core::Task {
 public:
  explicit CountOfSentSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string input_{};
  int count{};
};

class CountOfSentParallel : public ppc::core::Task {
 public:
  explicit CountOfSentParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::string input_{};
  std::string l_data{};
  int count{};
  boost::mpi::communicator world;
};

int CountOfSent(std::string& str, bool is_last = true);
} 