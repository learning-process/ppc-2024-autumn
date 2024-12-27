#pragma once

#include <cstdlib>
#include <ctime>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace nasedkin_e_seidels_iterate_methods_seq {

class SeidelIterateMethodsSEQ : public ppc::core::Task {
 public:
  explicit SeidelIterateMethodsSEQ(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void set_matrix(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vector);
  static void generate_random_matrix(int size, std::vector<std::vector<double>>& matrix, std::vector<double>& vector);
  const std::vector<double>& get_solution() const { return x; }
  double check_residual_norm() const;

 private:
  std::vector<std::vector<double>> A;
  std::vector<double> b;
  std::vector<double> x;
  int n;
  double epsilon;
  int max_iterations;

  bool converge(const std::vector<double>& x_new);
};

}  // namespace nasedkin_e_seidels_iterate_methods_seq