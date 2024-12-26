// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <memory>
#include <numbers>
#include <numeric>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace titov_s_global_optimization_2_mpi {
struct Point {
  double x;
  double y;
};

class MPIGlobalOpt2Sequential : public ppc::core::Task {
 public:
  explicit MPIGlobalOpt2Sequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
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

class MPIGlobalOpt2Parallel : public ppc::core::Task {
 public:
  explicit MPIGlobalOpt2Parallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double evaluate_function(const Point& point);
  bool is_within_bounds(const Point& point);
  bool all_constraints_satisfied(const Point& point);
  void calculate_initial_search_area();
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

  void split_search_area(int num_processes);
  void setup_mpi_operator();

  std::function<double(const Point&)> func_to_optimize_;
  std::vector<std::function<double(const Point&)>> constraints_funcs_;
  int max_iteration = 1000;
  size_t max_iterations_grad;
  double step_size;
  double tolerance;
  Point result_;
  double min_value_;
  double epsilon_ = 0.001;
  double lower_bound_x_;
  double upper_bound_x_;
  double lower_bound_y_;
  double upper_bound_y_;

  double process_lower_bound_x_;
  double process_upper_bound_x_;
  double process_lower_bound_y_;
  double process_upper_bound_y_;
  Point local_result_;
  Point local_initial_point_;
  double local_min_value_;
  std::vector<double> all_results_;

  boost::mpi::communicator world;
};

}  // namespace titov_s_global_optimization_2_mpi
