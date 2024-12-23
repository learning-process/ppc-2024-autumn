// mpi cpp rectangle method
#include "mpi/rezantseva_a_rectangle_method/include/ops_mpi_rez_a.hpp"

bool rezantseva_a_rectangle_method_mpi::check_integration_bounds(std::vector<std::pair<double, double>>* ib) {
  if (ib == nullptr) {
    std::cerr << "Error: bounds pointer is null." << std::endl;
    return false;
  }

  bool result = std::ranges::all_of(*ib, [](const auto& bound) { return bound.first < bound.second; });

  return result;
}

bool rezantseva_a_rectangle_method_mpi::RectangleMethodSequential::validation() {
  internal_order_test();
  auto* bounds = reinterpret_cast<std::vector<std::pair<double, double>>*>(taskData->inputs[0]);
  return (taskData->inputs.size() == 2 && taskData->outputs_count[0] == 1 &&
          (taskData->inputs_count[0] == taskData->inputs_count[1]) && check_integration_bounds(bounds));
}

bool rezantseva_a_rectangle_method_mpi::RectangleMethodSequential::pre_processing() {
  internal_order_test();

  auto* bounds = reinterpret_cast<std::vector<std::pair<double, double>>*>(taskData->inputs[0]);
  integration_bounds_.assign(bounds->begin(), bounds->end());

  auto* distribution_ptr = reinterpret_cast<std::vector<int>*>(taskData->inputs[1]);
  distribution_.assign(distribution_ptr->begin(), distribution_ptr->end());

  result_ = 0.0;

  return true;
}

bool rezantseva_a_rectangle_method_mpi::RectangleMethodSequential::run() {
  internal_order_test();
  int dimension = distribution_.size();
  std::vector<double> widths(dimension);
  int64_t rectangles = 1;

  for (int i = 0; i < dimension; i++) {
    widths[i] = (integration_bounds_[i].second - integration_bounds_[i].first) / static_cast<double>(distribution_[i]);
    rectangles *= distribution_[i];
  }
  std::vector<double> params(dimension);

  for (int i = 0; i < rectangles; i++) {
    int x = i;
    for (int j = 0; j < dimension; j++) {
      int idDimention = x % distribution_[j];
      params[j] = integration_bounds_[j].first + idDimention * widths[j] + widths[j] / 2;
      x /= distribution_[j];
    }
    result_ += func_(params);
  }
  for (int i = 0; i < dimension; i++) {
    result_ *= widths[i];
  }
  return true;
}

bool rezantseva_a_rectangle_method_mpi::RectangleMethodSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = result_;
  return true;
}

bool rezantseva_a_rectangle_method_mpi::RectangleMethodMPI::validation() {
  internal_order_test();
  bool flag = true;

  if (world.rank() == 0) {
    auto* bounds = reinterpret_cast<std::vector<std::pair<double, double>>*>(taskData->inputs[0]);
    flag = (taskData->inputs.size() == 2 && taskData->outputs_count[0] == 1 &&
            (taskData->inputs_count[0] == taskData->inputs_count[1]) && check_integration_bounds(bounds));
  }
  return flag;
}

bool rezantseva_a_rectangle_method_mpi::RectangleMethodMPI::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* bounds = reinterpret_cast<std::vector<std::pair<double, double>>*>(taskData->inputs[0]);
    integration_bounds_.assign(bounds->begin(), bounds->end());

    auto* distribution_ptr = reinterpret_cast<std::vector<int>*>(taskData->inputs[1]);
    distribution_.assign(distribution_ptr->begin(), distribution_ptr->end());

    num_processes_ = world.size();
    n_ = integration_bounds_.size();
    total_points_ = 1;
    widths_.assign(distribution_.size(), 0.0);
    for (int i = 0; i < n_; i++) {
      widths_[i] =
          (integration_bounds_[i].second - integration_bounds_[i].first) / static_cast<double>(distribution_[i]);
      if (i != n_ - 1) {
        total_points_ *= distribution_[i];
      }
    }
  }

  boost::mpi::broadcast(world, n_, 0);
  boost::mpi::broadcast(world, total_points_, 0);
  boost::mpi::broadcast(world, num_processes_, 0);
  boost::mpi::broadcast(world, distribution_, 0);
  boost::mpi::broadcast(world, widths_, 0);
  boost::mpi::broadcast(world, integration_bounds_, 0);

  result_ = 0.0;

  return true;
}

bool rezantseva_a_rectangle_method_mpi::RectangleMethodMPI::run() {
  internal_order_test();

  int remainder = total_points_ % num_processes_;
  int delta = total_points_ / num_processes_ + (world.rank() < remainder ? 1 : 0);

  std::vector<std::vector<double>> params(delta);

  int offset = 0;
  if (world.rank() < remainder) {
    offset = world.rank() * delta;
  } else {
    offset = remainder * (delta + 1) + (world.rank() - remainder) * delta;
  }

  for (int i = 0; i < delta; i++) {
    int x = offset + i;
    for (int j = 0; j < n_ - 1; j++) {
      int idDimention = x % distribution_[j];
      params[i].push_back(integration_bounds_[j].first + idDimention * widths_[j] + widths_[j] / 2);
      x /= distribution_[j];
    }
  }

  double local_sum = 0.0;
  for (int i = 0; i < delta; i++) {
    for (int j = 0; j < distribution_[n_ - 1]; j++) {
      params[i].push_back(integration_bounds_[n_ - 1].first + j * widths_[n_ - 1] + widths_[n_ - 1] / 2);
      local_sum += func_(params[i]);
      params[i].pop_back();
    }
  }

  boost::mpi::reduce(world, local_sum, result_, std::plus<>(), 0);
  for (int i = 0; i < n_; i++) {
    result_ *= widths_[i];
  }

  return true;
}

bool rezantseva_a_rectangle_method_mpi::RectangleMethodMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = result_;
  }
  return true;
}