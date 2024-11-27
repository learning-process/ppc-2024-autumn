#pragma once

#include <boost/mpi/communicator.hpp>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace nasedkin_e_seidels_iterate_methods_mpi {

class SeidelIterateMethodsMPI : public ppc::core::Task {
 public:
  explicit SeidelIterateMethodsMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  double compute_residual_norm();

 private:
  boost::mpi::communicator world;
  std::vector<std::vector<double>> A;
  std::vector<double> b;
  std::vector<double> x;
  int n;
  double epsilon;
  int max_iterations;

  bool converge(const std::vector<double>& x_new);
  void generate_valid_matrix();
};

}  // namespace nasedkin_e_seidels_iterate_methods_mpi
