#pragma once

#include <gtest/gtest.h>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "core/task/include/task.hpp"

namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_seq {

class MPIGaussianHorizontalSequential : public ppc::core::Task {
 public:
  explicit MPIGaussianHorizontalSequential(std::shared_ptr<ppc::core::TaskData> taskData_) 
      : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<double>> matrix;
  std::vector<double> vector_b;
};
}  // namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_seq