// Copyright 2023 Nesterov Alexander
#include "mpi/ermolaev_v_multidimensional_integral_rectangle/include/ops_mpi.hpp"

double ermolaev_v_multidimensional_integral_rectangle_mpi::integrateImpl(std::deque<std::pair<double, double>>& limits,
                                                                         std::vector<double>& args,
                                                                         const function& func, double eps) {
  double I = 0;
  double I0;
  int n = 2;

  auto [a, b] = limits.front();
  limits.pop_front();
  args.push_back(double{});

  do {
    I0 = I;
    I = 0;

    double h = (b - a) / n;
    args.back() = a + h / 2;
    for (int i = 0; i < n; i++) {
      if (limits.empty())
        I += func(args) * h;
      else
        I += ermolaev_v_multidimensional_integral_rectangle_mpi::integrateImpl(limits, args, func, eps) * h;

      args.back() += h;
    }

    n *= 2;
  } while (std::abs(I - I0) * 1 / 3 > eps);

  args.pop_back();
  limits.emplace_front(a, b);

  return I;
}

double ermolaev_v_multidimensional_integral_rectangle_mpi::integrateSeq(std::deque<std::pair<double, double>> limits,
                                                                        double eps, const function& func) {
  std::vector<double> args;
  return ermolaev_v_multidimensional_integral_rectangle_mpi::integrateImpl(limits, args, func, eps);
}

double ermolaev_v_multidimensional_integral_rectangle_mpi::integrateMPI(boost::mpi::communicator& world,
                                                                        std::deque<std::pair<double, double>> limits,
                                                                        double eps, const function& func) {
  std::vector<double> args;
  broadcast(world, limits, 0);
  broadcast(world, eps, 0);

  double I = 0;
  double I0;
  double localI;
  int n = world.size();

  auto [a, b] = limits.front();
  double step = (b - a) / world.size();

  double localA = a + step * world.rank();
  double localB = localA + step;

  limits.pop_front();
  args.push_back(double{});

  bool flag = true;
  while (flag) {
    I0 = I;
    I = 0;
    localI = 0;

    int count_of_points = n / world.size();
    double h = (localB - localA) / count_of_points;
    args.back() = localA + h / 2;
    for (int i = 0; i < count_of_points; i++) {
      if (limits.empty())
        localI += func(args) * h;
      else
        localI += integrateImpl(limits, args, func, eps) * h;

      args.back() += h;
    }

    reduce(world, localI, I, std::plus<>(), 0);

    if (world.rank() == 0 && std::abs(I - I0) * 1 / 3 <= eps) flag = false;
    broadcast(world, flag, 0);

    n *= 2;
  }

  if (world.rank() == 0) return I;
  return 0;
}

bool ermolaev_v_multidimensional_integral_rectangle_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  auto* ptr = reinterpret_cast<std::pair<double, double>*>(taskData->inputs[0]);
  limits_.assign(ptr, ptr + taskData->inputs_count[0]);
  eps_ = *reinterpret_cast<double*>(taskData->inputs[1]);

  return true;
}

bool ermolaev_v_multidimensional_integral_rectangle_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  return taskData->inputs_count[0] > 0 && taskData->inputs.size() == 2 && taskData->outputs_count[0] == 1 &&
         !taskData->outputs.empty();
}

bool ermolaev_v_multidimensional_integral_rectangle_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  res_ = integrateSeq(limits_, eps_, function_);

  return true;
}

bool ermolaev_v_multidimensional_integral_rectangle_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();

  reinterpret_cast<double*>(taskData->outputs[0])[0] = res_;

  return true;
}

bool ermolaev_v_multidimensional_integral_rectangle_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* ptr = reinterpret_cast<std::pair<double, double>*>(taskData->inputs[0]);
    limits_.assign(ptr, ptr + taskData->inputs_count[0]);
    eps_ = *reinterpret_cast<double*>(taskData->inputs[1]);
  }

  return true;
}

bool ermolaev_v_multidimensional_integral_rectangle_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  return (world.rank() != 0) || (taskData->inputs_count[0] > 0 && taskData->inputs.size() == 2 &&
                                 taskData->outputs_count[0] == 1 && !taskData->outputs.empty());
}

bool ermolaev_v_multidimensional_integral_rectangle_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  res_ = integrateMPI(world, limits_, eps_, function_);

  return true;
}

bool ermolaev_v_multidimensional_integral_rectangle_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = res_;
  }

  return true;
}
