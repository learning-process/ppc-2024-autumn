// Copyright 2024 Nesterov Alexander
#pragma once

#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <numbers>
#include <ranges>
#include <vector>

#include "core/task/include/task.hpp"

namespace titov_s_global_optimization_2_seq {

struct Point {
  double x;
  double y;
};

class GlobalOpt2Sequential : public ppc::core::Task {
 public:
  explicit GlobalOpt2Sequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double evaluate_function(const Point& point);
  bool is_within_bounds(const Point& point);
  bool all_constraints_satisfied(const Point& point);
  bool calculate_initial_search_area();
  Point compute_gradient(const Point& point);
  static Point Calculate(const Point& x, const Point& grad, double lambda);
  double GoldenSelection(double a, double b, double eps, const Point& grad, const Point& xj);
  double MakeSimplefx(double lambda, const Point& grad, const Point& xj);
  Point find_next_point(const Point& x_new);
  static Point project_on_constraint(const Point& point, const std::function<double(const Point&)>& constraint_func);
  static double compute_distance(const Point& p1, const Point& p2);
  static Point compute_constraint_gradient(const std::function<double(const Point&)>& constraint_func,
                                           const Point& point);
  static double evaluate_constraint(const std::function<double(const Point&)>& constraint_func, const Point& point);

  std::function<double(const Point&)> func_to_optimize_;
  std::vector<std::function<double(const Point&)>> constraints_funcs_;
  Point initial_point_;
  int max_iteration = 1000;
  Point result_;
  double min_value_;
  double epsilon_ = 0.1;

  double lower_bound_x_;
  double upper_bound_x_;
  double lower_bound_y_;
  double upper_bound_y_;
};

}  // namespace titov_s_global_optimization_2_seq
