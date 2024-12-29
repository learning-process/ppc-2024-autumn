#include "mpi/shulpin_i_simpson_method/include/simpson_method.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <cmath>
#include <functional>

double shulpin_simpson_method::f_x_plus_y(double x, double y) { return x + y; }
double shulpin_simpson_method::f_x_mul_y(double x, double y) { return x * y; }
double shulpin_simpson_method::f_sin_plus_cos(double x, double y) { return std::sin(x) + std::cos(y); }
double shulpin_simpson_method::f_sin_mul_cos(double x, double y) { return std::sin(x) * std::cos(y); }

inline double shulpin_simpson_method::calculate_coeff(const int* index, const int* limit) {
  if (*index == 0 || *index == *limit) {
    return 1.0;
  }
  return (*index % 2 == 0) ? 2.0 : 4.0;
}

double shulpin_simpson_method::calculate_row_sum(const int* i, const int* num_steps, const double* dx, const double* dy,
                                                 const double* a, const double* c, const func* func) {
  double row_sum = 0.0;
  double x = *a + (*i) * (*dx);
  double x_coeff = calculate_coeff(i, num_steps);

  for (int j = 0; j <= *num_steps; ++j) {
    double y = *c + j * (*dy);
    row_sum += x_coeff * calculate_coeff(&j, num_steps) * (*func)(x, y);
  }

  return row_sum;
}

double shulpin_simpson_method::seq_simpson(double a, double b, double c, double d, int N, const func& func_seq) {
  if (N % 2 != 0) {
    ++N;
  }

  double dx = (b - a) / N;
  double dy = (d - c) / N;
  double seq_sum = 0.0;

  for (int i = 0; i <= N; ++i) {
    seq_sum += calculate_row_sum(&i, &N, &dx, &dy, &a, &c, &func_seq);
  }

  return (dx * dy / 9.0) * seq_sum;
}

double shulpin_simpson_method::mpi_simpson(double a, double b, double c, double d, int N, const func& func_MPI) {
  boost::mpi::communicator world;
  int num_procs = world.size();
  int rank = world.rank();

  int num_steps = N;
  if (num_steps % 2 != 0) {
    ++num_steps;
  }

  double dx = (b - a) / num_steps;
  double dy = (d - c) / num_steps;

  int chunk_size = num_steps / num_procs;
  int extra_rows = num_steps % num_procs;

  int local_start = rank * chunk_size + std::min(rank, extra_rows);
  int local_end = local_start + chunk_size - 1;

  if (rank < extra_rows) {
    ++local_end;
  } else if (rank == num_procs - 1) {
    local_end = num_steps;
  }

  double local_sum = 0.0;

  for (int i = local_start; i <= local_end; ++i) {
    local_sum += calculate_row_sum(&i, &num_steps, &dx, &dy, &a, &c, &func_MPI);
  }

  double global_sum = 0.0;
  if (rank == 0) {
    global_sum = local_sum;
    for (int i = 1; i < num_procs; ++i) {
      double recv_sum;
      world.recv(i, 0, recv_sum);
      global_sum += recv_sum;
    }
    return (dx * dy / 9.0) * global_sum;
  }

  world.send(0, 0, local_sum);
  return 0.0;
}

bool shulpin_simpson_method::SimpsonMethodSeq::pre_processing() {
  internal_order_test();

  double a_value = *reinterpret_cast<double*>(taskData->inputs[0]);
  double b_value = *reinterpret_cast<double*>(taskData->inputs[1]);
  double c_value = *reinterpret_cast<double*>(taskData->inputs[2]);
  double d_value = *reinterpret_cast<double*>(taskData->inputs[3]);
  int N_value = *reinterpret_cast<int*>(taskData->inputs[4]);

  a_seq = a_value;
  b_seq = b_value;
  c_seq = c_value;
  d_seq = d_value;
  N_seq = N_value;

  return true;
}

bool shulpin_simpson_method::SimpsonMethodSeq::validation() {
  internal_order_test();

  return ((taskData->inputs.size() == 5) && (*reinterpret_cast<int*>(taskData->inputs[4]) > 0) &&
          (*reinterpret_cast<double*>(taskData->inputs[0]) < *reinterpret_cast<double*>(taskData->inputs[1])) &&
          (*reinterpret_cast<double*>(taskData->inputs[2]) < *reinterpret_cast<double*>(taskData->inputs[3])));
}

bool shulpin_simpson_method::SimpsonMethodSeq::run() {
  internal_order_test();

  res_seq = seq_simpson(a_seq, b_seq, c_seq, d_seq, N_seq, func_seq);

  return true;
}

bool shulpin_simpson_method::SimpsonMethodSeq::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res_seq;

  return true;
}

bool shulpin_simpson_method::SimpsonMethodMPI::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    double a_value = *reinterpret_cast<double*>(taskData->inputs[0]);
    double b_value = *reinterpret_cast<double*>(taskData->inputs[1]);
    double c_value = *reinterpret_cast<double*>(taskData->inputs[2]);
    double d_value = *reinterpret_cast<double*>(taskData->inputs[3]);
    int N_value = *reinterpret_cast<int*>(taskData->inputs[4]);

    a_MPI = a_value;
    b_MPI = b_value;
    c_MPI = c_value;
    d_MPI = d_value;
    N_MPI = N_value;
  }

  return true;
}

bool shulpin_simpson_method::SimpsonMethodMPI::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return ((taskData->inputs.size() == 5) && (*reinterpret_cast<int*>(taskData->inputs[4]) > 0) &&
            (*reinterpret_cast<double*>(taskData->inputs[0]) < *reinterpret_cast<double*>(taskData->inputs[1])) &&
            (*reinterpret_cast<double*>(taskData->inputs[2]) < *reinterpret_cast<double*>(taskData->inputs[3])));
  }

  return true;
}

bool shulpin_simpson_method::SimpsonMethodMPI::run() {
  internal_order_test();

  double local_res{};

  boost::mpi::broadcast(world, a_MPI, 0);
  boost::mpi::broadcast(world, b_MPI, 0);
  boost::mpi::broadcast(world, c_MPI, 0);
  boost::mpi::broadcast(world, d_MPI, 0);
  boost::mpi::broadcast(world, N_MPI, 0);

  local_res = mpi_simpson(a_MPI, b_MPI, c_MPI, d_MPI, N_MPI, func_MPI);

  boost::mpi::reduce(world, local_res, res_MPI, std::plus<>(), 0);
  return true;
}

bool shulpin_simpson_method::SimpsonMethodMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = res_MPI;
  }

  return true;
}

void shulpin_simpson_method::SimpsonMethodSeq::set_seq(const func& f) { func_seq = f; }

void shulpin_simpson_method::SimpsonMethodMPI::set_MPI(const func& f) { func_MPI = f; }