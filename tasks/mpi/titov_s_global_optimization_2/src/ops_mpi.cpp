// Copyright 2023 Nesterov Alexander
#include "mpi/titov_s_global_optimization_2/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"

bool titov_s_global_optimization_2_mpi::MPIGlobalOpt2Sequential::pre_processing() {
  internal_order_test();

  auto* func_ptr = reinterpret_cast<std::function<double(const Point&)>*>(taskData->inputs[0]);
  func_to_optimize_ = *func_ptr;

  auto* constraints_ptr = reinterpret_cast<std::vector<std::function<double(const Point&)>>*>(taskData->inputs[1]);

  constraints_funcs_ = *constraints_ptr;

  min_value_ = std::numeric_limits<double>::infinity();
  result_ = {0.0, 0.0};

  return calculate_initial_search_area();
}
bool titov_s_global_optimization_2_mpi::MPIGlobalOpt2Sequential::validation() {
  internal_order_test();
  return true;
}
bool titov_s_global_optimization_2_mpi::MPIGlobalOpt2Sequential::calculate_initial_search_area() {
  double test_range = 10.0;
  double step = 0.1;

  lower_bound_x_ = std::numeric_limits<double>::infinity();
  upper_bound_x_ = -std::numeric_limits<double>::infinity();
  lower_bound_y_ = std::numeric_limits<double>::infinity();
  upper_bound_y_ = -std::numeric_limits<double>::infinity();

  Point initial_point{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
  int num_steps = static_cast<int>(2 * test_range / step) + 1;

  for (int i = 0; i < num_steps; ++i) {
    double x = -test_range + i * step;
    for (int j = 0; j < num_steps; ++j) {
      double y = -test_range + j * step;

      Point test_point{x, y};
      bool satisfies_all_constraints = true;
      for (const auto& constraint : constraints_funcs_) {
        if (constraint(test_point) <= 0) {
          satisfies_all_constraints = false;
          break;
        }
      }

      if (satisfies_all_constraints) {
        lower_bound_x_ = std::min(lower_bound_x_, test_point.x);
        upper_bound_x_ = std::max(upper_bound_x_, test_point.x);
        lower_bound_y_ = std::min(lower_bound_y_, test_point.y);
        upper_bound_y_ = std::max(upper_bound_y_, test_point.y);

        if (initial_point.x == std::numeric_limits<double>::infinity() &&
            initial_point.y == std::numeric_limits<double>::infinity()) {
          initial_point = test_point;
        }
      }
    }
  }

  if (initial_point.x != std::numeric_limits<double>::infinity() &&
      initial_point.y != std::numeric_limits<double>::infinity()) {
    initial_point_ = initial_point;
  } else {
    return false;
  }

  lower_bound_x_ -= 1;
  upper_bound_x_ += 1;
  lower_bound_y_ -= 1;
  upper_bound_y_ += 1;
  return true;
}

bool titov_s_global_optimization_2_mpi::MPIGlobalOpt2Sequential::run() {
  internal_order_test();

  result_ = initial_point_;

  min_value_ = evaluate_function(result_);

  for (int iteration = 0; iteration < max_iteration; ++iteration) {
    Point grad = compute_gradient(result_);
    double lambda = GoldenSelection(0, 0.1, epsilon_, grad, result_);

    Point grad_point = Calculate(result_, grad, lambda);
    Point new_point = find_next_point(grad_point);

    double new_value = evaluate_function(new_point);
    double curent_value = min_value_;
    if (new_value < min_value_) {
      min_value_ = new_value;
      result_ = new_point;
    }

    if (std::abs(new_value - curent_value) < epsilon_) {
      break;
    }
  }

  return true;
}
titov_s_global_optimization_2_mpi::Point titov_s_global_optimization_2_mpi::MPIGlobalOpt2Sequential::compute_gradient(
    const Point& point) {
  const double h = 1e-5;
  double fx_val = evaluate_function(point);

  return {(evaluate_function({point.x + h, point.y}) - fx_val) / h,
          (evaluate_function({point.x, point.y + h}) - fx_val) / h};
}
titov_s_global_optimization_2_mpi::Point titov_s_global_optimization_2_mpi::MPIGlobalOpt2Sequential::Calculate(
    const Point& x, const Point& grad, double lambda) {
  return {x.x - lambda * grad.x, x.y - lambda * grad.y};
}
double titov_s_global_optimization_2_mpi::MPIGlobalOpt2Sequential::GoldenSelection(double a, double b, double eps,
                                                                                   const Point& grad, const Point& xj) {
  const double phi = std::numbers::phi;
  double x1;
  double x2;
  double y1;
  double y2;

  x1 = b - (b - a) / phi;
  x2 = a + (b - a) / phi;
  y1 = MakeSimplefx(x1, grad, xj);
  y2 = MakeSimplefx(x2, grad, xj);

  while (std::abs(b - a) > eps) {
    if (y1 <= y2) {
      b = x2;
      x2 = x1;
      x1 = b - (b - a) / phi;
      y2 = y1;
      y1 = MakeSimplefx(x1, grad, xj);
    } else {
      a = x1;
      x1 = x2;
      x2 = a + (b - a) / phi;
      y1 = y2;
      y2 = MakeSimplefx(x2, grad, xj);
    }
  }

  return (a + b) / 2;
}
double titov_s_global_optimization_2_mpi::MPIGlobalOpt2Sequential::MakeSimplefx(double lambda, const Point& grad,
                                                                                const Point& xj) {
  Point buffer = {xj.x - lambda * grad.x, xj.y - lambda * grad.y};
  return evaluate_function(buffer);
}
titov_s_global_optimization_2_mpi::Point titov_s_global_optimization_2_mpi::MPIGlobalOpt2Sequential::find_next_point(
    const Point& x_new) {
  Point current_point = x_new;
  double step_size = 0.5;
  double tolerance = 0.0001;
  size_t max_iterations = 100;

  for (size_t iteration = 0; iteration < max_iterations; ++iteration) {
    Point correction;
    correction.x = 0;
    correction.y = 0;
    bool constraints_violated = false;

    for (size_t i = 0; i < constraints_funcs_.size(); ++i) {
      double violation = constraints_funcs_[i](current_point);
      if (violation < 0) {
        constraints_violated = true;

        Point proj = project_on_constraint(current_point, constraints_funcs_[i]);
        correction.x += proj.x - current_point.x;
        correction.y += proj.y - current_point.y;
      }
    }

    if (!constraints_violated) {
      return current_point;
    }

    current_point.x += step_size * correction.x;
    current_point.y += step_size * correction.y;

    if (std::abs(correction.x) < tolerance && std::abs(correction.y) < tolerance) {
      return current_point;
    }
  }
  return current_point;
}
titov_s_global_optimization_2_mpi::Point
titov_s_global_optimization_2_mpi::MPIGlobalOpt2Sequential::project_on_constraint(
    const Point& point, const std::function<double(const Point&)>& constraint_func) {
  Point grad = compute_constraint_gradient(constraint_func, point);
  double g_val = evaluate_constraint(constraint_func, point);

  Point proj;
  proj.x = point.x - g_val * grad.x;
  proj.y = point.y - g_val * grad.y;

  return proj;
}
titov_s_global_optimization_2_mpi::Point
titov_s_global_optimization_2_mpi::MPIGlobalOpt2Sequential::compute_constraint_gradient(
    const std::function<double(const Point&)>& constraint_func, const Point& point) {
  double h = 1e-5;

  Point grad;
  Point point_dx = point;
  Point point_dy = point;

  point_dx.x += h;
  point_dy.y += h;

  double constraint_dx = evaluate_constraint(constraint_func, point_dx) - evaluate_constraint(constraint_func, point);
  double constraint_dy = evaluate_constraint(constraint_func, point_dy) - evaluate_constraint(constraint_func, point);

  grad.x = constraint_dx / h;
  grad.y = constraint_dy / h;

  return grad;
}
double titov_s_global_optimization_2_mpi::MPIGlobalOpt2Sequential::evaluate_constraint(
    const std::function<double(const Point&)>& constraint_func, const Point& point) {
  double result = constraint_func(point);
  return result;
}
double titov_s_global_optimization_2_mpi::MPIGlobalOpt2Sequential::compute_distance(const Point& p1, const Point& p2) {
  return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}
bool titov_s_global_optimization_2_mpi::MPIGlobalOpt2Sequential::post_processing() {
  internal_order_test();
  reinterpret_cast<Point*>(taskData->outputs[0])[0] = result_;
  return true;
}
double titov_s_global_optimization_2_mpi::MPIGlobalOpt2Sequential::evaluate_function(const Point& point) {
  return func_to_optimize_(point);
}
bool titov_s_global_optimization_2_mpi::MPIGlobalOpt2Sequential::all_constraints_satisfied(const Point& point) {
  return std::ranges::all_of(constraints_funcs_, [&](const auto& constraint) { return constraint(point) > 0; });
}

bool titov_s_global_optimization_2_mpi::MPIGlobalOpt2Parallel::pre_processing() {
  internal_order_test();
  auto* func_ptr = reinterpret_cast<std::function<double(const Point&)>*>(taskData->inputs[0]);
  func_to_optimize_ = *func_ptr;
  auto* constraints_ptr = reinterpret_cast<std::vector<std::function<double(const Point&)>>*>(taskData->inputs[1]);
  constraints_funcs_ = *constraints_ptr;
  step_size = *reinterpret_cast<double*>(taskData->inputs[2]);
  tolerance = *reinterpret_cast<double*>(taskData->inputs[3]);
  max_iterations_grad = *reinterpret_cast<size_t*>(taskData->inputs[4]);
  if (world.rank() == 0) {
    calculate_initial_search_area();
  }

  min_value_ = std::numeric_limits<double>::infinity();
  result_ = {0.0, 0.0};

  return true;
}

bool titov_s_global_optimization_2_mpi::MPIGlobalOpt2Parallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs[0] == nullptr || taskData->inputs[1] == nullptr || taskData->inputs.empty() ||
        taskData->inputs_count.size() < 2) {
      return false;
    }
    auto* func_ptr = reinterpret_cast<std::function<double(const Point&)>*>(taskData->inputs[0]);
    auto* constraints_ptr = reinterpret_cast<std::vector<std::function<bool(const Point&)>>*>(taskData->inputs[1]);
    if (func_ptr == nullptr || constraints_ptr == nullptr) {
      return false;
    }

    if (taskData->outputs.empty() || taskData->outputs_count.size() != 1) {
      return false;
    }
    return true;
  }
  return true;
}

bool titov_s_global_optimization_2_mpi::MPIGlobalOpt2Parallel::run() {
  internal_order_test();

  split_search_area(world.size());

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist_x(process_lower_bound_x_, process_upper_bound_x_);
  std::uniform_real_distribution<double> dist_y(process_lower_bound_y_, process_upper_bound_y_);

  do {
    local_initial_point_.x = dist_x(gen);
    local_initial_point_.y = dist_y(gen);
  } while (!all_constraints_satisfied(local_initial_point_));

  local_result_ = local_initial_point_;
  local_min_value_ = evaluate_function(local_result_);
  for (int iteration = 0; iteration < max_iteration; ++iteration) {
    Point grad = compute_gradient(local_result_);
    double lambda = GoldenSelection(0, 0.1, epsilon_, grad, local_result_);
    Point grad_point = Calculate(local_result_, grad, lambda);
    Point new_point = find_next_point(grad_point);
    double new_value = evaluate_function(new_point);
    double current_value = local_min_value_;

    if (new_value < local_min_value_) {
      local_min_value_ = new_value;
      local_result_ = new_point;
    }

    if (std::abs(new_value - current_value) < epsilon_) {
      break;
    }
  }

  std::vector<double> local_coordinates = {local_result_.x, local_result_.y};

  std::vector<int> sizes(world.size(), 2);
  all_results_.resize(2 * world.size());
  gatherv(world, local_coordinates.data(), local_coordinates.size(), all_results_.data(), sizes, 0);

  if (world.rank() == 0) {
    for (size_t i = 0; i < all_results_.size(); i += 2) {
      Point point{all_results_[i], all_results_[i + 1]};
      double value = evaluate_function(point);
      if (value < min_value_) {
        min_value_ = value;
        result_ = point;
      }
    }
  }

  return true;
}

void titov_s_global_optimization_2_mpi::MPIGlobalOpt2Parallel::split_search_area(int num_processes) {
  std::vector<double> all_bounds_flat(num_processes * 4);

  if (world.rank() == 0) {
    for (int rank = 0; rank < num_processes; ++rank) {
      double y_step = (upper_bound_y_ - lower_bound_y_) / num_processes;

      double process_lower_bound_y = lower_bound_y_ + rank * y_step;
      double process_upper_bound_y = process_lower_bound_y + y_step;

      double process_lower_bound_x = lower_bound_x_;
      double process_upper_bound_x = upper_bound_x_;

      all_bounds_flat[rank * 4] = process_lower_bound_x;
      all_bounds_flat[rank * 4 + 1] = process_upper_bound_x;
      all_bounds_flat[rank * 4 + 2] = process_lower_bound_y;
      all_bounds_flat[rank * 4 + 3] = process_upper_bound_y;
    }
  }

  std::vector<int> send_counts(num_processes, 4);
  std::vector<int> displacements(num_processes, 0);
  for (int i = 1; i < num_processes; ++i) {
    displacements[i] = displacements[i - 1] + send_counts[i - 1];
  }

  std::vector<double> bounds(4);
  boost::mpi::scatterv(world, all_bounds_flat.data(), send_counts, displacements, bounds.data(), 4, 0);

  process_lower_bound_x_ = bounds[0];
  process_upper_bound_x_ = bounds[1];
  process_lower_bound_y_ = bounds[2];
  process_upper_bound_y_ = bounds[3];
}

bool titov_s_global_optimization_2_mpi::MPIGlobalOpt2Parallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<Point*>(taskData->outputs[0])[0] = {result_.x, result_.y};
  }
  return true;
}

void titov_s_global_optimization_2_mpi::MPIGlobalOpt2Parallel::calculate_initial_search_area() {
  double test_range = 10.0;
  double step = 0.01;

  lower_bound_x_ = std::numeric_limits<double>::infinity();
  upper_bound_x_ = -std::numeric_limits<double>::infinity();
  lower_bound_y_ = std::numeric_limits<double>::infinity();
  upper_bound_y_ = -std::numeric_limits<double>::infinity();

  int num_steps = static_cast<int>(2 * test_range / step) + 1;

  for (int i = 0; i < num_steps; ++i) {
    double x = -test_range + i * step;
    for (int j = 0; j < num_steps; ++j) {
      double y = -test_range + j * step;
      Point test_point{x, y};

      if (all_constraints_satisfied(test_point)) {
        lower_bound_x_ = std::min(lower_bound_x_, test_point.x);
        upper_bound_x_ = std::max(upper_bound_x_, test_point.x);
        lower_bound_y_ = std::min(lower_bound_y_, test_point.y);
        upper_bound_y_ = std::max(upper_bound_y_, test_point.y);
      }
    }
  }

  lower_bound_x_ -= 0.1;
  upper_bound_x_ += 0.1;
  lower_bound_y_ -= 0.1;
  upper_bound_y_ += 0.1;
}

titov_s_global_optimization_2_mpi::Point titov_s_global_optimization_2_mpi::MPIGlobalOpt2Parallel::compute_gradient(
    const Point& point) {
  const double h = 1e-5;
  double fx_val = evaluate_function(point);

  return {(evaluate_function({point.x + h, point.y}) - fx_val) / h,
          (evaluate_function({point.x, point.y + h}) - fx_val) / h};
}

titov_s_global_optimization_2_mpi::Point titov_s_global_optimization_2_mpi::MPIGlobalOpt2Parallel::Calculate(
    const Point& x, const Point& grad, double lambda) {
  return {x.x - lambda * grad.x, x.y - lambda * grad.y};
}

double titov_s_global_optimization_2_mpi::MPIGlobalOpt2Parallel::GoldenSelection(double a, double b, double eps,
                                                                                 const Point& grad, const Point& xj) {
  const double phi = std::numbers::phi;
  double x1;
  double x2;
  double y1;
  double y2;

  x1 = b - (b - a) / phi;
  x2 = a + (b - a) / phi;
  y1 = MakeSimplefx(x1, grad, xj);
  y2 = MakeSimplefx(x2, grad, xj);

  while (std::abs(b - a) > eps) {
    if (y1 <= y2) {
      b = x2;
      x2 = x1;
      x1 = b - (b - a) / phi;
      y2 = y1;
      y1 = MakeSimplefx(x1, grad, xj);
    } else {
      a = x1;
      x1 = x2;
      x2 = a + (b - a) / phi;
      y1 = y2;
      y2 = MakeSimplefx(x2, grad, xj);
    }
  }

  return (a + b) / 2;
}

double titov_s_global_optimization_2_mpi::MPIGlobalOpt2Parallel::MakeSimplefx(double lambda, const Point& grad,
                                                                              const Point& xj) {
  Point buffer = {xj.x - lambda * grad.x, xj.y - lambda * grad.y};
  return evaluate_function(buffer);
}

titov_s_global_optimization_2_mpi::Point titov_s_global_optimization_2_mpi::MPIGlobalOpt2Parallel::find_next_point(
    const Point& x_new) {
  Point current_point = x_new;

  auto local_constraints = constraints_funcs_;
  local_constraints.emplace_back([this](const Point& p) { return process_upper_bound_x_ - p.x; });
  local_constraints.emplace_back([this](const Point& p) { return p.x - process_lower_bound_x_; });
  local_constraints.emplace_back([this](const Point& p) { return process_upper_bound_y_ - p.y; });
  local_constraints.emplace_back([this](const Point& p) { return p.y - process_lower_bound_y_; });

  for (size_t iteration = 0; iteration < max_iterations_grad; ++iteration) {
    Point correction;
    correction.x = 0;
    correction.y = 0;
    bool constraints_violated = false;

    for (size_t i = 0; i < local_constraints.size(); ++i) {
      double violation = local_constraints[i](current_point);
      if (violation <= 0) {
        constraints_violated = true;

        Point proj = project_on_constraint(current_point, local_constraints[i]);
        correction.x += proj.x - current_point.x;
        correction.y += proj.y - current_point.y;
      }
    }

    if (!constraints_violated) {
      return current_point;
    }

    current_point.x += step_size * correction.x;
    current_point.y += step_size * correction.y;

    if (std::abs(correction.x) < tolerance && std::abs(correction.y) < tolerance) {
      return current_point;
    }
  }

  return current_point;
}

titov_s_global_optimization_2_mpi::Point
titov_s_global_optimization_2_mpi::MPIGlobalOpt2Parallel::project_on_constraint(
    const Point& point, const std::function<double(const Point&)>& constraint_func) {
  Point grad = compute_constraint_gradient(constraint_func, point);
  double g_val = evaluate_constraint(constraint_func, point);

  Point proj;
  proj.x = point.x - g_val * grad.x;
  proj.y = point.y - g_val * grad.y;

  return proj;
}

titov_s_global_optimization_2_mpi::Point
titov_s_global_optimization_2_mpi::MPIGlobalOpt2Parallel::compute_constraint_gradient(
    const std::function<double(const Point&)>& constraint_func, const Point& point) {
  double h = 1e-5;

  Point grad;
  Point point_dx = point;
  Point point_dy = point;

  point_dx.x += h;
  point_dy.y += h;

  double constraint_dx = evaluate_constraint(constraint_func, point_dx) - evaluate_constraint(constraint_func, point);
  double constraint_dy = evaluate_constraint(constraint_func, point_dy) - evaluate_constraint(constraint_func, point);

  grad.x = constraint_dx / h;
  grad.y = constraint_dy / h;

  return grad;
}

double titov_s_global_optimization_2_mpi::MPIGlobalOpt2Parallel::evaluate_constraint(
    const std::function<double(const Point&)>& constraint_func, const Point& point) {
  double result = constraint_func(point);
  return result;
}

double titov_s_global_optimization_2_mpi::MPIGlobalOpt2Parallel::compute_distance(const Point& p1, const Point& p2) {
  return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

double titov_s_global_optimization_2_mpi::MPIGlobalOpt2Parallel::evaluate_function(const Point& point) {
  return func_to_optimize_(point);
}

bool titov_s_global_optimization_2_mpi::MPIGlobalOpt2Parallel::all_constraints_satisfied(const Point& point) {
  return std::ranges::all_of(constraints_funcs_, [&](const auto& constraint) { return constraint(point) > 0; });
}
