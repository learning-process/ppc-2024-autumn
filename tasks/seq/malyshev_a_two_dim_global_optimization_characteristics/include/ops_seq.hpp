// Copyright 2023 Nesterov Alexander
#pragma once
#define _USE_MATH_DEFINES

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace malyshev_a_two_dim_global_optimization_characteristics_seq {

using constraint_t = std::function<bool(double, double)>;
using target_t = std::function<double(double, double)>;

struct Point {
  double x;
  double y;
  double value;

  Point(double x = 0, double y = 0) : x(x), y(y) { value = std::numeric_limits<double>::max(); }
  Point(double x, double y, double value) : x(x), y(y), value(value) {}
};

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_, target_t traget,
                              std::vector<constraint_t> constraints)
      : Task(std::move(taskData_)), traget_function_(std::move(traget)), constraints_(std::move(constraints)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 protected:
  struct Constants {
    static constexpr double h = 1e-7;
    static constexpr int max_iterations = 100;
    static constexpr int grid_initial_size = 20;
    static constexpr double tunnel_rate = 0.1;
    static constexpr double start_learning_rate = 0.01;
    static constexpr int num_tunnels = 20;
  };

  struct Data {
    double x_min;
    double x_max;
    double y_min;
    double y_max;
    double eps;
  } data_;

  target_t traget_function_;
  std::vector<constraint_t> constraints_;
  Point res_;

  virtual Point local_search(double x0, double y0);
  virtual Point tunnel_search(const Point& current_min);
  virtual bool check_constraints(double x, double y);

  void readTaskData();
  void writeTaskData();
  bool validateTaskData();
  virtual void optimize();
};

}  // namespace malyshev_a_two_dim_global_optimization_characteristics_seq
