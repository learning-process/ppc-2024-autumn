#pragma once

#include <boost/mpi.hpp>
#include <functional>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace chernykh_a_multidimensional_integral_simpson_mpi {

using bound_t = std::pair<double, double>;
using bounds_t = std::vector<bound_t>;
using step_range_t = std::pair<int, int>;
using func_args_t = std::vector<double>;
using func_1d_t = std::function<double(double)>;
using func_nd_t = std::function<double(const func_args_t &)>;

double integrate_1d(const func_1d_t &func, const bound_t &bound, int num_steps);

double integrate_nd(const func_nd_t &func, func_args_t &func_args, const bounds_t &bounds,
                    const step_range_t &step_range, double tolerance, int dim);

class SequentialTask : public ppc::core::Task {
 public:
  explicit SequentialTask(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  func_nd_t func;
  func_args_t func_args;
  bounds_t bounds;
  step_range_t step_range;
  double tolerance{};
  double result{};
};

class ParallelTask : public ppc::core::Task {
 public:
  explicit ParallelTask(std::shared_ptr<ppc::core::TaskData> task_data, func_nd_t &func)
      : Task(std::move(task_data)), func(func) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  func_nd_t func;
  func_args_t func_args;
  bounds_t bounds;
  step_range_t step_range;
  double tolerance{};
  double result{};
  boost::mpi::communicator world;

  bound_t get_chunk_bound(int dim) const;
};

}  // namespace chernykh_a_multidimensional_integral_simpson_mpi
