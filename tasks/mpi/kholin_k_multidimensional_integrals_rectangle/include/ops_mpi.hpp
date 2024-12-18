#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace enum_ops {
enum operations { MULTISTEP_SCHEME_METHOD_RECTANGLE };
};

namespace kholin_k_multidimensional_integrals_rectangle_mpi {
using Function = std::function<double(const std::vector<double>&)>;

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_, enum_ops::operations ops_)
      : Task(std::move(taskData_)), ops(std::move(ops_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> f_values;
  Function f;
  std::vector<double> lower_limits;
  std::vector<double> upper_limits;
  double epsilon;
  double result;

  size_t dim;
  size_t sz_values;
  size_t sz_lower_limits;
  size_t sz_upper_limits;

  double integrate(Function f_, const std::vector<double>& l_limits, const std::vector<double>& u_limits,
                   const std::vector<double>& h, std::vector<double>& f_values_, size_t curr_index_dim, size_t dim_,
                   size_t n);
  double integrate_with_rectangle_method(Function f_, std::vector<double>& f_values_,
                                         const std::vector<double>& l_limits, const std::vector<double>& u_limits,
                                         size_t dim_, size_t n);
  double run_multistep_scheme_method_rectangle(Function f_, std::vector<double>& f_values_,
                                               const std::vector<double>& l_limits, const std::vector<double>& u_limits,
                                               size_t dim_, double epsilon_);
  enum_ops::operations ops;
};

class TestMPITaskParallel : public ppc::core::Task {
  MPI_Datatype get_mpi_type();

 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_, Function f_)
      : Task(std::move(taskData_)), f(f_) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  ~TestMPITaskParallel();

 private:
  std::vector<double> f_values;
  Function f;
  std::vector<double> lower_limits;
  std::vector<double> upper_limits;
  double epsilon;

  size_t dim;
  size_t sz_values;
  size_t sz_lower_limits;
  size_t sz_upper_limits;

  std::vector<double> local_l_limits;
  std::vector<double> local_u_limits;
  double I_2n;

  double integrate(Function f_, const std::vector<double>& l_limits, const std::vector<double>& u_limits,
                   const std::vector<double>& h, std::vector<double>& f_values_, size_t curr_index_dim, size_t dim_,
                   size_t n);
  double integrate_with_rectangle_method(Function f_, std::vector<double>& f_values_,
                                         const std::vector<double>& l_limits, const std::vector<double>& u_limits,
                                         size_t dim_, size_t n);
  double run_multistep_scheme_method_rectangle(Function f_, std::vector<double>& f_values_,
                                               const std::vector<double>& l_limits, const std::vector<double>& u_limits,
                                               size_t dim_, double epsilon_);
  MPI_Datatype sz_t;
};

}  // namespace kholin_k_multidimensional_integrals_rectangle_mpi