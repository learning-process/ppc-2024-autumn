
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

namespace vershinina_a_image_smoothing {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  int rows{};
  int cols{};

 private:
  std::vector<int> output_;
  int *input_{};
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  int rows{};
  int local_rows{};
  int cols{};
  int global_res{};
  int rows_per_proc;
  int remainder;
  std::vector<int> local_input_;
  std::vector<int> local_output_;
  std::vector<int> local_input_sizes;

 private:
  std::vector<int> output_;
  std::vector<int> input_;
  boost::mpi::communicator world;
};

}  // namespace vershinina_a_image_smoothing