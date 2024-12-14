#include "mpi/malyshev_a_two_dim_global_optimization_characteristics/include/ops_mpi.hpp"

bool malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskSequential::pre_processing() {
  internal_order_test();

  readTaskData();

  return true;
}

bool malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskSequential::validation() {
  internal_order_test();

  return validateTaskData();
}

bool malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskSequential::run() {
  internal_order_test();

  optimize();

  return true;
}

bool malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskSequential::post_processing() {
  internal_order_test();

  writeTaskData();

  return true;
}

void malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskSequential::readTaskData() {
  auto* bounds_ptr = reinterpret_cast<std::pair<double, double>*>(taskData->inputs[0]);
  auto x_pair = *bounds_ptr;
  data_.x_min = x_pair.first;
  data_.x_max = x_pair.second;

  auto y_pair = *(bounds_ptr + 1);
  data_.y_min = y_pair.first;
  data_.y_max = y_pair.second;

  auto* eps_ptr = reinterpret_cast<double*>(taskData->inputs[1]);
  data_.eps = *eps_ptr;
}

bool malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskSequential::validateTaskData() {
  return taskData != nullptr && taskData->inputs.size() >= 2 && !taskData->outputs.empty() &&
         taskData->inputs[0] != nullptr && taskData->inputs[1] != nullptr && taskData->outputs[0] != nullptr &&
         (*reinterpret_cast<double*>(taskData->inputs[1]) > 0);
}

void malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskSequential::writeTaskData() {
  *reinterpret_cast<Point*>(taskData->outputs[0]) = res_;
}

void malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskSequential::optimize() {
  double x_step = (data_.x_max - data_.x_min) / Constants::grid_initial_size;
  double y_step = (data_.y_max - data_.y_min) / Constants::grid_initial_size;
  Point best_point(0, 0, std::numeric_limits<double>::max());

  double x;
  double y;
  for (int i = 0; i <= Constants::grid_initial_size; i++) {
    for (int j = 0; j <= Constants::grid_initial_size; j++) {
      x = data_.x_min + i * x_step;
      y = data_.y_min + j * y_step;

      Point local_min = local_search(x, y);

      if (local_min.value < best_point.value) {
        best_point = local_min;
      }
    }
  }

  int no_improvement_count = 0;
  for (int i = 0; i < Constants::max_iterations && no_improvement_count < 10; i++) {
    Point tunneled_point = tunnel_search(best_point);

    if (tunneled_point.value < best_point.value - data_.eps) {
      best_point = tunneled_point;
      no_improvement_count = 0;
    } else {
      no_improvement_count++;
    }
  }

  res_ = local_search(best_point.x, best_point.y);
}

malyshev_a_two_dim_global_optimization_characteristics_mpi::Point
malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskSequential::local_search(double x0, double y0) {
  Point current(x0, y0);
  current.value = traget_function_(x0, y0);

  double learning_rate = Constants::start_learning_rate;
  const double learning_rate_decay = 0.95;

  double dx = (traget_function_(x0 + Constants::h, y0) - traget_function_(x0 - Constants::h, y0)) / (2 * Constants::h);
  double dy = (traget_function_(x0, y0 + Constants::h) - traget_function_(x0, y0 - Constants::h)) / (2 * Constants::h);

  double new_x;
  double new_y;
  for (int i = 0; i < Constants::max_iterations; i++) {
    new_x = current.x - learning_rate * dx;
    new_y = current.y - learning_rate * dy;

    if (new_x < data_.x_min || new_x > data_.x_max || new_y < data_.y_min || new_y > data_.y_max ||
        !check_constraints(new_x, new_y)) {
      learning_rate *= learning_rate_decay;
      continue;
    }

    double new_value = traget_function_(new_x, new_y);

    if (new_value < current.value) {
      current.x = new_x;
      current.y = new_y;
      current.value = new_value;
    } else {
      learning_rate *= learning_rate_decay;
    }

    double grad_norm = std::sqrt(dx * dx + dy * dy);
    if (grad_norm < data_.eps) {
      break;
    }
  }

  if (!check_constraints(current.x, current.y)) {
    return Point(current.x, current.y, std::numeric_limits<double>::max());
  }

  return current;
}

malyshev_a_two_dim_global_optimization_characteristics_mpi::Point
malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskSequential::tunnel_search(
    const Point& current_min) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  double radius = std::min(data_.x_max - data_.x_min, data_.y_max - data_.y_min) * Constants::tunnel_rate;
  Point best_point = current_min;

  double angle;
  double r;
  double new_x;
  double new_y;
  for (int i = 0; i < Constants::num_tunnels; i++) {
    angle = 2 * M_PI * dis(gen);
    r = radius * std::sqrt(std::abs(dis(gen)));

    new_x = current_min.x + r * std::cos(angle);
    new_y = current_min.y + r * std::sin(angle);

    if (new_x >= data_.x_min && new_x <= data_.x_max && new_y >= data_.y_min && new_y <= data_.y_max) {
      Point new_point = local_search(new_x, new_y);
      if (new_point.value < best_point.value) {
        best_point = new_point;
      }
    }
  }
  return best_point;
}

bool malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskSequential::check_constraints(double x,
                                                                                                       double y) {
  return std::all_of(constraints_.begin(), constraints_.end(),
                     [x, y](const constraint_t& constraint) { return constraint(x, y); });
}

bool malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) readTaskData();
  return true;
}

bool malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskParallel::validation() {
  internal_order_test();

  return (world.rank() != 0) || validateTaskData();
}

bool malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskParallel::run() {
  internal_order_test();

  broadcast(world, data_, 0);
  distrebute_constraints();
  optimize();

  return true;
}

bool malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) writeTaskData();
  return true;
}

malyshev_a_two_dim_global_optimization_characteristics_mpi::Point
malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskParallel::tunnel_search(const Point& current_min) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  double radius = std::min(data_.x_max - data_.x_min, data_.y_max - data_.y_min) * Constants::tunnel_rate;
  Point best_point = current_min;

  std::vector<Point> new_points(Constants::num_tunnels);
  if (world.rank() == 0) {
    double angle;
    double r;
    double new_x;
    double new_y;
    for (int i = 0; i < Constants::num_tunnels; i++) {
      angle = 2 * M_PI * dis(gen);
      r = radius * std::sqrt(std::abs(dis(gen)));
      new_x = current_min.x + r * std::cos(angle);
      new_y = current_min.y + r * std::sin(angle);
      new_points[i] = Point(new_x, new_y, std::numeric_limits<double>::max());
    }
  }

  broadcast(world, new_points, 0);

  for (int i = 0; i < Constants::num_tunnels; i++) {
    if (new_points[i].x >= data_.x_min && new_points[i].x <= data_.x_max && new_points[i].y >= data_.y_min &&
        new_points[i].y <= data_.y_max) {
      Point new_point = local_search(new_points[i].x, new_points[i].y);
      if (new_point.value < best_point.value) {
        best_point = new_point;
      }
    }
  }
  return best_point;
}

bool malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskParallel::check_constraints(double x,
                                                                                                     double y) {
  bool local_result = true;
  if (is_active_)
    local_result = std::all_of(constraints_.begin() + start_constraint_index_,
                               constraints_.begin() + start_constraint_index_ + local_constraint_count_,
                               [x, y](const constraint_t& constraint) { return constraint(x, y); });

  MPI_Win_fence(0, bool_win);
  shared_results[world.rank()] = local_result;
  MPI_Win_fence(0, bool_win);

  bool global_result = true;
  for (int i = 0; i < world.size(); ++i) {
    if (!shared_results[i]) {
      global_result = false;
      break;
    }
  }

  return global_result;
}

void malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskParallel::distrebute_constraints() {
  int delta_constraints = constraints_.size() / world.size();
  int extra_constraints = constraints_.size() % world.size();
  std::vector<int> local_constraint_sizes(world.size(), delta_constraints);
  for (int i = 0; i < extra_constraints; i++) {
    local_constraint_sizes[world.size() - 1 - i]++;
  }

  local_constraint_count_ = local_constraint_sizes[world.rank()];
  is_active_ = local_constraint_count_ != 0;
  start_constraint_index_ =
      std::accumulate(local_constraint_sizes.begin(), local_constraint_sizes.begin() + world.rank(), 0);
}

void malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskParallel::init_shared_bool_array() {
  MPI_Aint size = (world.rank() == 0) ? world.size() * sizeof(bool) : 0;
  int disp_unit = sizeof(bool);

  MPI_Win_allocate_shared(size, disp_unit, MPI_INFO_NULL, world, &shared_results, &bool_win);

  if (world.rank() != 0) {
    MPI_Aint win_size;
    int win_disp;

    MPI_Win_shared_query(bool_win, 0, &win_size, &win_disp, reinterpret_cast<void**>(&shared_results));
  } else {
    std::fill_n(shared_results, world.size(), false);
  }

  MPI_Win_fence(MPI_MODE_NOPRECEDE, bool_win);
}

void malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskParallel::cleanup_shared_arrays() {
  MPI_Win_fence(MPI_MODE_NOSUCCEED, bool_win);
  MPI_Win_free(&bool_win);
}

malyshev_a_two_dim_global_optimization_characteristics_mpi::Point
malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskParallel::local_search(double x0, double y0) {
  const auto diff_x = [this](double x, double y) {
    return (traget_function_(x + Constants::h, y) - traget_function_(x - Constants::h, y)) / (2 * Constants::h);
  };

  const auto diff_y = [this](double x, double y) {
    return (traget_function_(x, y + Constants::h) - traget_function_(x, y - Constants::h)) / (2 * Constants::h);
  };

  Point current(x0, y0);
  current.value = traget_function_(x0, y0);

  double learning_rate = Constants::start_learning_rate;
  int iterations_without_improvement = 0;
  const int max_without_improvement = 5;
  const double learning_rate_decay = 0.95;
  const double min_learning_rate = 1e-8;

  double dx = diff_x(x0, y0);
  double dy = diff_y(x0, y0);

  double new_x;
  double new_y;
  for (int i = 0; i < Constants::max_iterations / 2; i++) {
    if (iterations_without_improvement >= max_without_improvement || learning_rate < min_learning_rate) {
      break;
    }

    new_x = current.x - learning_rate * dx;
    new_y = current.y - learning_rate * dy;

    if (new_x < data_.x_min || new_x > data_.x_max || new_y < data_.y_min || new_y > data_.y_max ||
        !check_constraints(new_x, new_y)) {
      learning_rate *= learning_rate_decay;
      iterations_without_improvement++;
      continue;
    }

    double new_value = traget_function_(new_x, new_y);

    if (new_value < current.value) {
      current.x = new_x;
      current.y = new_y;
      current.value = new_value;

      dx = diff_x(new_x, new_y);
      dy = diff_y(new_x, new_y);

      iterations_without_improvement = 0;
    } else {
      learning_rate *= learning_rate_decay;
      iterations_without_improvement++;
    }

    double grad_norm = std::sqrt(dx * dx + dy * dy);
    if (grad_norm < min_learning_rate) {
      break;
    }
  }

  if (!check_constraints(current.x, current.y)) {
    return Point(current.x, current.y, std::numeric_limits<double>::max());
  }

  return current;
}
