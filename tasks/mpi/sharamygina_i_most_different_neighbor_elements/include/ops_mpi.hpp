#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <climits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sharamygina_i_most_different_neighbor_elements_mpi {

class most_different_neighbor_elements_seq : public ppc::core::Task {
 public:
  explicit most_different_neighbor_elements_seq(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  int res{};
};

class most_different_neighbor_elements_mpi : public ppc::core::Task {
 public:
  explicit most_different_neighbor_elements_mpi(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_, local_input_;
  int res;
  size_t size;
  size_t st;
  boost::mpi::communicator world;
};

}  // namespace sharamygina_i_most_different_neighbor_elements_mpi
