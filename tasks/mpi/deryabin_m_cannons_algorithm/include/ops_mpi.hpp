#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

#include "core/task/include/task.hpp"

namespace deryabin_m_cannons_algorithm_mpi {

class CannonsAlgorithmMPITaskSequential : public ppc::core::Task {
 public:
  explicit CannonsAlgorithmMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> input_matrix_A;
  std::vector<double> input_matrix_B;
  std::vector<double> output_matrix_C;
};
class CannonsAlgorithmMPITaskParallel : public ppc::core::Task {
 public:
  explicit CannonsAlgorithmMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> input_matrix_A, local_input_matrix_A;
  std::vector<double> input_matrix_B, local_input_matrix_B;
  std::vector<double> output_matrix_C, local_output_matrix_C;
  boost::mpi::communicator world;
};
}  // namespace deryabin_m_cannons_algorithm_mpi
