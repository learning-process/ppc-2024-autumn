// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace beresnev_a_increase_contrast_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  size_t pixel_data_size{}, pixel_data_start{};
  size_t width{}, height{}, max_color{};
  double factor{};
  std::vector<uint8_t> inp_, res_;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  size_t pixel_data_size{}, pixel_data_start{};
  size_t width{}, height{}, max_color{};
  size_t delta{}, rem{};
  double factor{};
  std::vector<uint8_t> inp_, res_;
  boost::mpi::communicator world;
};

}  // namespace beresnev_a_increase_contrast_mpi