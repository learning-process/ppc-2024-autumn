#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace plekhanov_d_verticalgaus_mpi {

void copy_input_data(double* source, std::vector<double>& destination, size_t size);

bool validate_image(const std::vector<double>& image, size_t width, size_t height);

void apply_gaussian_filter(const std::vector<double>& input, std::vector<double>& output, size_t width, size_t height);

class VerticalGausSeqTest : public ppc::core::Task {
 public:
  explicit VerticalGausSeqTest(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> inputImage, outputImage;
  int inputWidth{}, inputHeight{};
};

class VerticalGausMPITest : public ppc::core::Task {
 public:
  explicit VerticalGausMPITest(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> inputImage, outputImage;
  int inputWidth{}, inputHeight{};
  boost::mpi::communicator world;
};

}  // namespace plekhanov_d_verticalgaus_mpi