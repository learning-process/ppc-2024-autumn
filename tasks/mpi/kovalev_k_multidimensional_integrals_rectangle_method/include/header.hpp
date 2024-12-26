#pragma once

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stack>
#include <vector>

#include "core/task/include/task.hpp"

namespace kovalev_k_multidimensional_integrals_rectangle_method_mpi {
class MultidimensionalIntegralsRectangleMethodPar : public ppc::core::Task {
 private:
  std::vector<std::pair<double, double>> limits;
  size_t n;
  std::function<double(std::vector<double>& args)> func;
  double h;
  double l_res;
  double g_res;
  boost::mpi::communicator world;

 public:
  explicit MultidimensionalIntegralsRectangleMethodPar(std::shared_ptr<ppc::core::TaskData> taskData_,
                                                       std::function<double(std::vector<double>& args)> func_)
      : Task(std::move(taskData_)), func(std::move(func_)) {}
  bool count_multidimensional_integrals_rectangle_method_mpi();
  double customRound(double value) const;
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
};
}  // namespace kovalev_k_multidimensional_integrals_rectangle_method_mpi