#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

#include "core/task/include/task.hpp"

namespace deryabin_m_jacobi_iterative_method_mpi {

class JacobiIterativeMPITaskSequential : public ppc::core::Task {
 public:
  explicit JacobiIterativeMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> input_matrix_;
  std::vector<double> input_right_vector_;
  std::vector<double> output_x_vector_;
};
class JacobiIterativeMPITaskParallel : public ppc::core::Task {
 public:
  explicit JacobiIterativeMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> input_matrix_, local_input_matrix_part_;
  std::vector<double> input_right_vector_, local_input_right_vector_part_;
  std::vector<double> output_x_vector_, local_output_x_vector_part_;
  boost::mpi::communicator world;
};
}  // namespace deryabin_m_jacobi_iterative_method_mpi
