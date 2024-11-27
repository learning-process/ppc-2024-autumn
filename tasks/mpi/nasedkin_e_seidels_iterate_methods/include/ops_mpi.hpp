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
  void set_matrix_and_vector(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vector);

 private:
  boost::mpi::communicator world;
  std::vector<std::vector<double>> A;
  std::vector<double> b;
  std::vector<double> x;
  int n;
  double epsilon;
  int max_iterations;
  bool validate_result();

  bool converge(const std::vector<double>& x_new);
};

}  // namespace nasedkin_e_seidels_iterate_methods_mpi