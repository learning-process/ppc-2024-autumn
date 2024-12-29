// Copyright 2023 Nesterov Alexander
#include "mpi/prokhorov_n_global_search_algorithm_strongin/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <vector>

namespace prokhorov_n_global_search_algorithm_strongin_mpi {

double stronginAlgorithmImpl(double& a, double& b, double epsilon, const std::function<double(double)>& f) {
  double x_min = a;
  double f_min = f(x_min);

  while ((b - a) > epsilon) {
    double x1 = a + (b - a) / 3.0;
    double x2 = b - (b - a) / 3.0;

    double f1 = f(x1);
    double f2 = f(x2);

    if (f1 < f2) {
      b = x2;
      if (f1 < f_min) {
        f_min = f1;
        x_min = x1;
      }
    } else {
      a = x1;
      if (f2 < f_min) {
        f_min = f2;
        x_min = x2;
      }
    }
  }

  return x_min;
}

double prokhorov_n_global_search_algorithm_strongin_mpi::TestMPITaskSequential::stronginAlgorithm() {
  return stronginAlgorithmImpl(a, b, epsilon, f);
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  a = reinterpret_cast<double*>(taskData->inputs[0])[0];
  b = reinterpret_cast<double*>(taskData->inputs[1])[0];
  epsilon = reinterpret_cast<double*>(taskData->inputs[2])[0];

  f = [](double x) { return x * x; };

  result = 0.0;
  return true;
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  return taskData->inputs_count[0] == 1 && taskData->inputs_count[1] == 1 && taskData->inputs_count[2] == 1 &&
         taskData->outputs_count[0] == 1;
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  result = stronginAlgorithm();
  return true;
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();

  reinterpret_cast<double*>(taskData->outputs[0])[0] = result;
  return true;
}

double prokhorov_n_global_search_algorithm_strongin_mpi::TestMPITaskParallel::stronginAlgorithm() {
  return stronginAlgorithmImpl(a, b, epsilon, f);
}

double prokhorov_n_global_search_algorithm_strongin_mpi::TestMPITaskParallel::stronginAlgorithmParallel() {
  double global_min = std::numeric_limits<double>::max();
  double global_x_min = a;

  while ((b - a) > epsilon) {
    double local_min = std::numeric_limits<double>::max();
    double local_x_min;

    double step = (b - a) / world.size();
    double local_a = a + step * world.rank();
    double local_b = a + step * (world.rank() + 1);

    double x1 = local_a + (local_b - local_a) / 3.0;
    double x2 = local_b - (local_b - local_a) / 3.0;

    double f1 = f(x1);
    double f2 = f(x2);

    if (f1 < f2) {
      local_min = f1;
      local_x_min = x1;
    } else {
      local_min = f2;
      local_x_min = x2;
    }

    std::vector<double> all_mins(world.size());
    std::vector<double> all_x_mins(world.size());
    gather(world, local_min, all_mins, 0);
    gather(world, local_x_min, all_x_mins, 0);

    world.barrier();

    if (world.rank() == 0) {
      for (int i = 0; i < world.size(); ++i) {
        if (all_mins[i] < global_min) {
          global_min = all_mins[i];
          global_x_min = all_x_mins[i];
        }
      }

      a = global_x_min - step;
      b = global_x_min + step;
    }

    broadcast(world, a, 0);
    broadcast(world, b, 0);
  }

  return global_x_min;
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData->inputs.empty() || taskData->inputs.size() < 3) {
      throw std::runtime_error("Not enough input data.");
    }

    if (taskData->inputs[0] == nullptr || taskData->inputs[1] == nullptr || taskData->inputs[2] == nullptr) {
      throw std::runtime_error("Input data is null.");
    }

    a = *reinterpret_cast<double*>(taskData->inputs[0]);
    b = *reinterpret_cast<double*>(taskData->inputs[1]);
    epsilon = *reinterpret_cast<double*>(taskData->inputs[2]);
  }

  broadcast(world, a, 0);
  broadcast(world, b, 0);
  broadcast(world, epsilon, 0);

  if (world.rank() != 0) {
    if (a == 0.0 && b == 0.0 && epsilon == 0.0) {
      throw std::runtime_error("Data was not broadcasted correctly.");
    }
  }

  f = [](double x) { return x * x; };

  result = 0.0;
  return true;
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  return taskData->inputs_count[0] == 1 && taskData->inputs_count[1] == 1 && taskData->inputs_count[2] == 1 &&
         taskData->outputs_count[0] == 1;
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  if (world.size() == 1) {
    result = stronginAlgorithm();
  } else {
    result = stronginAlgorithmParallel();
  }

  return true;
}

bool prokhorov_n_global_search_algorithm_strongin_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = result;
  }
  return true;
}

}  // namespace prokhorov_n_global_search_algorithm_strongin_mpi