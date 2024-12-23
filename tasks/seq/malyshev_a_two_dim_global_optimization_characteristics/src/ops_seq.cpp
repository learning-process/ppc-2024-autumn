#include "seq/malyshev_a_two_dim_global_optimization_characteristics/include/ops_seq.hpp"

bool malyshev_a_two_dim_global_optimization_characteristics_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  readTaskData();

  return true;
}

bool malyshev_a_two_dim_global_optimization_characteristics_seq::TestTaskSequential::validation() {
  internal_order_test();

  return validateTaskData();
}

bool malyshev_a_two_dim_global_optimization_characteristics_seq::TestTaskSequential::run() {
  internal_order_test();

  optimize();

  return true;
}

bool malyshev_a_two_dim_global_optimization_characteristics_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  writeTaskData();

  return true;
}

void malyshev_a_two_dim_global_optimization_characteristics_seq::TestTaskSequential::readTaskData() {
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

bool malyshev_a_two_dim_global_optimization_characteristics_seq::TestTaskSequential::validateTaskData() {
  return taskData != nullptr && taskData->inputs.size() >= 2 && !taskData->outputs.empty() &&
         taskData->inputs[0] != nullptr && taskData->inputs[1] != nullptr && taskData->outputs[0] != nullptr &&
         (*reinterpret_cast<double*>(taskData->inputs[1]) > 0);
}

void malyshev_a_two_dim_global_optimization_characteristics_seq::TestTaskSequential::writeTaskData() {
  *reinterpret_cast<Point*>(taskData->outputs[0]) = res_;
}

void malyshev_a_two_dim_global_optimization_characteristics_seq::TestTaskSequential::optimize() {
  double x_step = (data_.x_max - data_.x_min) / Constants::grid_initial_size;
  double y_step = (data_.y_max - data_.y_min) / Constants::grid_initial_size;
  Point best_point(0, 0, std::numeric_limits<double>::max());

  for (int i = 0; i <= Constants::grid_initial_size; i++) {
    for (int j = 0; j <= Constants::grid_initial_size; j++) {
      double x = data_.x_min + i * x_step;
      double y = data_.y_min + j * y_step;

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

malyshev_a_two_dim_global_optimization_characteristics_seq::Point
malyshev_a_two_dim_global_optimization_characteristics_seq::TestTaskSequential::local_search(double x0, double y0) {
  Point current(x0, y0);
  current.value = traget_function_(x0, y0);

  double learning_rate = Constants::start_learning_rate;
  const double learning_rate_decay = 0.95;

  double dx = (traget_function_(x0 + Constants::h, y0) - traget_function_(x0 - Constants::h, y0)) / (2 * Constants::h);
  double dy = (traget_function_(x0, y0 + Constants::h) - traget_function_(x0, y0 - Constants::h)) / (2 * Constants::h);

  for (int i = 0; i < Constants::max_iterations; i++) {
    double new_x = current.x - learning_rate * dx;
    double new_y = current.y - learning_rate * dy;

    if (new_x < data_.x_min || new_x > data_.x_max || new_y < data_.y_min || new_y > data_.y_max) {
      learning_rate *= learning_rate_decay;
      continue;
    }

    if (!check_constraints(new_x, new_y)) {
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
malyshev_a_two_dim_global_optimization_characteristics_seq::Point
malyshev_a_two_dim_global_optimization_characteristics_seq::TestTaskSequential::tunnel_search(
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

bool malyshev_a_two_dim_global_optimization_characteristics_seq::TestTaskSequential::check_constraints(double x,
                                                                                                       double y) {
  return std::all_of(constraints_.begin(), constraints_.end(),
                     [x, y](const constraint_t& constraint) { return constraint(x, y); });
}
