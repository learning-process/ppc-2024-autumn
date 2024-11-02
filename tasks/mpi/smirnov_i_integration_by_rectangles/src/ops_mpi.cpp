#include "mpi/smirnov_i_integration_by_rectangles/include/ops_mpi.hpp"

#include <algorithm>
#include <exception>
#include <functional>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool smirnov_i_integration_by_rectangles::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    left = reinterpret_cast<double*>(taskData->inputs[0])[0];
    right = reinterpret_cast<double*>(taskData->inputs[1])[0];
    n_ = reinterpret_cast<int*>(taskData->inputs[2])[0];   
  }

  broadcast(world, left, 0);
  broadcast(world, right, 0);
  broadcast(world, n_, 0);
  return true;
}
bool smirnov_i_integration_by_rectangles::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }

  return true;
}
bool smirnov_i_integration_by_rectangles::TestMPITaskParallel::run() {
  internal_order_test();
  double local_result_{};
  local_result_ = mpi_integrate_rect(f, left, right, n_);
  reduce(world, local_result_, glob_res, std::plus<>(), 0);
  return true;
}
bool smirnov_i_integration_by_rectangles::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = glob_res;
  }
  return true;
}
void smirnov_i_integration_by_rectangles::TestMPITaskParallel::set_function(double (*func)(double)) { f = func; }

double smirnov_i_integration_by_rectangles::TestMPITaskParallel::mpi_integrate_rect(double (*func)(double),
                                                                                    double left_, double right_,
                                                                                    int n) {
  if (func == nullptr) {
    throw std::logic_error("func is nullptr");
  }
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int chunks = n / size;
  int dop = n % size;
  if (dop != 0) {
    if (rank < dop) {
      chunks++;
    }
  }
  double res_integr = 0;

  if (rank == 0) {
    const double self_left = left_;
    const double self_right = left_ + (right_ - left_) / size;
    const double len_of_rect = (self_right - self_left) / chunks;
    for (int i = 0; i < chunks; i++) {
      const double left_rect = self_left + i * len_of_rect;
      res_integr += f(left_rect + len_of_rect / 2);
    }
    res_integr *= len_of_rect;
    double recv;
    for (int i = 1; i < size; i++) {
      MPI_Recv(&recv, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
      res_integr += recv;
    }
  } else {
    const double gap_for_proc = (right_ - left_) / size;
    double self_res_integr = 0;
    const double self_left = left_ + gap_for_proc * rank;
    const double self_right = left_ + gap_for_proc * (rank + 1);
    const double len_of_rect = (self_right - self_left) / chunks;
    for (int i = 0; i < chunks; i++) {
      const double left_rect = self_left + i * len_of_rect;
      self_res_integr += f(left_rect + len_of_rect / 2);
    }
    self_res_integr *= len_of_rect;
    MPI_Send(&self_res_integr, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
  
  return res_integr;
}

bool smirnov_i_integration_by_rectangles::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  left = reinterpret_cast<double*>(taskData->inputs[0])[0];
  right = reinterpret_cast<double*>(taskData->inputs[1])[0];
  n_ = reinterpret_cast<int*>(taskData->inputs[2])[0];
  res = 0;
  return true;
}

bool smirnov_i_integration_by_rectangles::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}
bool smirnov_i_integration_by_rectangles::TestMPITaskSequential::run() {
  internal_order_test();
  res = seq_integrate_rect(f, left, right, n_);
  return true;
}
bool smirnov_i_integration_by_rectangles::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}
void smirnov_i_integration_by_rectangles::TestMPITaskSequential::set_function(double (*func)(double)) { f = func; }

double smirnov_i_integration_by_rectangles::TestMPITaskSequential::seq_integrate_rect(double (*func)(double),
                                                                                      double left_, double right_,
                                                                                      int n) {
  if (func == nullptr) {
    throw std::logic_error("func is nullptr");
  }
  double res_integr = 0;
  const double self_left = left_;
  const double self_right = right_;
  const double len_of_rect = (self_right - self_left) / n;
  for (int i = 0; i < n; i++) {
    const double left_rect = self_left + i * len_of_rect;
    res_integr += f(left_rect + len_of_rect / 2);
  }
  res_integr *= len_of_rect;
  return res_integr;
}
