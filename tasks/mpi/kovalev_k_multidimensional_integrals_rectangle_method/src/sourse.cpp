#include "mpi/kovalev_k_multidimensional_integrals_rectangle_method/include/header.hpp"

double
kovalev_k_multidimensional_integrals_rectangle_method_mpi::MultidimensionalIntegralsRectangleMethodPar::customRound(
    double value) const {
  int tmp = static_cast<int>(1 / h);
  int decimalPlaces = 0;
  while (tmp > 0 && tmp % 10 == 0) {
    decimalPlaces++;
    tmp /= 10;
  }

  double factor = std::pow(10.0, decimalPlaces);
  return std::round(value * factor) / factor;
}

bool kovalev_k_multidimensional_integrals_rectangle_method_mpi::MultidimensionalIntegralsRectangleMethodPar::
    count_multidimensional_integrals_rectangle_method_mpi() {
  std::stack<std::vector<double>> stack;
  stack.emplace();

  while (!stack.empty()) {
    std::vector<double> point = stack.top();
    stack.pop();

    if (point.size() == n) {
      l_res += func(point) * std::pow(h, n);
      continue;
    }

    int dim = point.size();

    for (double x = limits[dim].first; x + h <= limits[dim].second; x += h) {
      point.push_back(x + h / 2);
      stack.emplace(point);
      point.pop_back();
    }
  }

  return true;
}

bool kovalev_k_multidimensional_integrals_rectangle_method_mpi::MultidimensionalIntegralsRectangleMethodPar::
    pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    n = taskData->inputs_count[0];
    limits.resize(n);
    void* ptr_input = taskData->inputs[0];
    void* ptr_vec = limits.data();
    h = reinterpret_cast<double*>(taskData->inputs[1])[0];
    memcpy(ptr_vec, ptr_input, sizeof(std::pair<double, double>) * n);
  }

  g_res = l_res = 0;
  return true;
}

bool kovalev_k_multidimensional_integrals_rectangle_method_mpi::MultidimensionalIntegralsRectangleMethodPar::
    validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs.empty() || taskData->outputs.empty() || taskData->inputs_count[0] <= 0 ||
        taskData->outputs_count[0] != 1 || reinterpret_cast<double*>(taskData->inputs[1])[0] > 0.01) {
      return false;
    }
  }
  return true;
}

bool kovalev_k_multidimensional_integrals_rectangle_method_mpi::MultidimensionalIntegralsRectangleMethodPar::run() {
  internal_order_test();
  boost::mpi::broadcast(world, n, 0);
  boost::mpi::broadcast(world, h, 0);

  if (world.rank() != 0) {
    limits.resize(n);
  }

  MPI_Bcast(limits.data(), n * sizeof(std::pair<double, double>), MPI_BYTE, 0, MPI_COMM_WORLD);

  double delta = limits[0].second / world.size() - limits[0].first / world.size();

  limits[0].first = customRound(limits[0].first + delta * world.rank());
  limits[0].second = customRound(limits[0].second - delta * (world.size() - world.rank() - 1));

  count_multidimensional_integrals_rectangle_method_mpi();

  boost::mpi::reduce(world, l_res, g_res, std::plus<>(), 0);

  return true;
}

bool kovalev_k_multidimensional_integrals_rectangle_method_mpi::MultidimensionalIntegralsRectangleMethodPar::
    post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = g_res;
  }
  return true;
}