// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <utility>

#include "core/task/include/task.hpp"

namespace zolotareva_a_smoothing_image_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  static std::vector<float> create_gaussian_kernel(int radius, float sigma);
  static void convolve_rows(const std::vector<uint8_t>& input, int height, int width, const std::vector<float>& kernel,
                            std::vector<float>& temp);
  static void convolve_columns(const std::vector<float>& temp, int height, int width, const std::vector<float>& kernel,
                               std::vector<uint8_t>& output);

 private:
  std::vector<uint8_t> input_;
  std::vector<uint8_t> result_;
  int width_{0};
  int height_{0};
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<uint8_t> input_;
  std::vector<uint8_t> result_;
  std::vector<uint8_t> local_input_;
  std::vector<int> send_counts;
  int width_{0};
  int height_{0};
  int size_{0};
  int local_height_{0};
  boost::mpi::communicator world;
};

}  // namespace zolotareva_a_smoothing_image_mpi