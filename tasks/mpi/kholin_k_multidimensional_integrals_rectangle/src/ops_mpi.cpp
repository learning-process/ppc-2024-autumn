#include "mpi/kholin_k_multidimensional_integrals_rectangle/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

double kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential::integrate(
    Function f_, const std::vector<double>& l_limits, const std::vector<double>& u_limits, const std::vector<double>& h,
    std::vector<double>& f_values_, size_t curr_index_dim, size_t dim_, size_t n) {
  if (curr_index_dim == dim_) {
    return f_(f_values_);
  }

  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    f_values_[curr_index_dim] = l_limits[curr_index_dim] + (i + 0.5) * h[curr_index_dim];
    sum += integrate(f_, l_limits, u_limits, h, f_values_, curr_index_dim + 1, dim_, n);
  }
  return sum * h[curr_index_dim];
}

double kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential::integrate_with_rectangle_method(
    Function f_, std::vector<double>& f_values_, const std::vector<double>& l_limits,
    const std::vector<double>& u_limits, size_t dim_, size_t n) {
  std::vector<double> h(dim_);
  for (size_t i = 0; i < dim_; ++i) {
    h[i] = (u_limits[i] - l_limits[i]) / n;
  }

  return integrate(f_, l_limits, u_limits, h, f_values_, 0, dim_, n);
}

double kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential::run_multistep_scheme_method_rectangle(
    Function f_, std::vector<double>& f_values_, const std::vector<double>& l_limits,
    const std::vector<double>& u_limits, size_t dim_, double epsilon_) {
  int n = 1;
  double I_n = integrate_with_rectangle_method(f_, f_values_, l_limits, u_limits, dim_, n);
  double I_2n;
  double delta = 0;
  do {
    n *= 2;
    I_2n = integrate_with_rectangle_method(f_, f_values_, l_limits, u_limits, dim_, n);
    delta = std::abs(I_2n - I_n);
    I_n = I_2n;

    /*   std::cout << "n: " << n << ", I: " << I_2n << ", delta : " << delta << std::endl;*/

  } while ((1.0 / 3) * delta >= epsilon_);

  return I_2n;
}

bool kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  sz_values = taskData->inputs_count[0];
  sz_lower_limits = taskData->inputs_count[1];
  sz_upper_limits = taskData->inputs_count[2];

  auto ptr_dim = reinterpret_cast<size_t*>(taskData->inputs[0]);
  dim = *ptr_dim;

  auto* ptr_f_values = reinterpret_cast<double*>(taskData->inputs[1]);
  f_values.assign(ptr_f_values, ptr_f_values + sz_values);

  auto ptr_f = reinterpret_cast<std::function<double(const std::vector<double>&)>*>(taskData->inputs[2]);
  f = *ptr_f;

  auto* ptr_lower_limits = reinterpret_cast<double*>(taskData->inputs[3]);
  lower_limits.assign(ptr_lower_limits, ptr_lower_limits + sz_lower_limits);

  auto* ptr_upper_limits = reinterpret_cast<double*>(taskData->inputs[4]);
  upper_limits.assign(ptr_upper_limits, ptr_upper_limits + sz_upper_limits);

  auto ptr_epsilon = reinterpret_cast<double*>(taskData->inputs[5]);
  epsilon = *ptr_epsilon;

  result = 0.0;
  return true;
}

bool kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[1] && taskData->inputs_count[2];
}

bool kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  if (ops == enum_ops::MULTISTEP_SCHEME_METHOD_RECTANGLE) {
    // std::cout << "seq" << std::endl;
    result = run_multistep_scheme_method_rectangle(f, f_values, lower_limits, upper_limits, dim, epsilon);
  }
  return true;
}

bool kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = result;
  return true;
}

double kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel::integrate(
    Function f_, const std::vector<double>& l_limits, const std::vector<double>& u_limits, const std::vector<double>& h,
    std::vector<double>& f_values_, size_t curr_index_dim, size_t dim_, size_t n) {
  if (curr_index_dim == dim_) {
    return f_(f_values_);
  }

  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    f_values_[curr_index_dim] = l_limits[curr_index_dim] + (i + 0.5) * h[curr_index_dim];
    sum += integrate(f_, l_limits, u_limits, h, f_values_, curr_index_dim + 1, dim_, n);
  }
  return sum * h[curr_index_dim];
}

double kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel::integrate_with_rectangle_method(
    Function f_, std::vector<double>& f_values_, const std::vector<double>& l_limits,
    const std::vector<double>& u_limits, size_t dim_, size_t n) {
  std::vector<double> h(dim_);
  for (size_t i = 0; i < dim_; ++i) {
    h[i] = (u_limits[i] - l_limits[i]) / n;
  }

  return integrate(f_, l_limits, u_limits, h, f_values_, 0, dim_, n);
}

double kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel::run_multistep_scheme_method_rectangle(
    Function f_, std::vector<double>& f_values_, const std::vector<double>& l_limits,
    const std::vector<double>& u_limits, size_t dim_, double epsilon_) {
  int size, ProcRank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

  int n = 1;
  double I_n = 0.0;
  if (ProcRank == 0) {
    I_n = integrate_with_rectangle_method(f_, f_values_, l_limits, u_limits, dim_, n);
  }
  MPI_Bcast(&I_n, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  double local_result;
  double delta = 0;

  if (ProcRank >= 0) {
    local_l_limits = std::vector<double>(dim_);
    local_u_limits = std::vector<double>(dim_);
  }

  for (size_t i = 0; i < dim_; ++i) {
    double range = u_limits[i] - l_limits[i];

    local_l_limits[i] = l_limits[i] + (ProcRank * (range / size));
    local_u_limits[i] = l_limits[i] + ((ProcRank + 1) * (range / size));
  }

  // std::cout << "Im process " << ProcRank << std::endl;
  // std::cout << "my local_lower_limits is ";
  // for (size_t i = 0; i < sz_lower_limits; i++) {
  //   std::cout << " " << local_l_limits[i];
  // }
  // std::cout << std::endl;
  // std::cout << "my local_upper_limits is ";
  // for (size_t i = 0; i < sz_upper_limits; i++) {
  //   std::cout << " " << local_u_limits[i];
  // }
  // std::cout << std::endl;

  do {
    n *= 2;

    local_result = integrate_with_rectangle_method(f_, f_values_, local_l_limits, local_u_limits, dim_, n);
    if (dim_ > 1) local_result = local_result * std::pow(size, dim - 1);
    MPI_Reduce(&local_result, &I_2n, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (ProcRank == 0) {
      delta = std::abs(I_2n - I_n);
      I_n = I_2n;
    }
    MPI_Bcast(&delta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /*   if (ProcRank == 0) {
         std::cout << "n: " << n << ", I: " << I_2n << ", delta : " << delta << std::endl;
       }*/

  } while ((1.0 / 3) * delta >= epsilon_);

  return I_2n;
}

bool kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  // Init value for input and output
  int size, ProcRank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  if (ProcRank == 0) {
    sz_values = taskData->inputs_count[0];
    sz_lower_limits = taskData->inputs_count[1];
    sz_upper_limits = taskData->inputs_count[2];

    auto ptr_dim = reinterpret_cast<size_t*>(taskData->inputs[0]);
    dim = *ptr_dim;

    auto* ptr_f_values = reinterpret_cast<double*>(taskData->inputs[1]);
    f_values.assign(ptr_f_values, ptr_f_values + sz_values);

    auto* ptr_lower_limits = reinterpret_cast<double*>(taskData->inputs[2]);
    lower_limits.assign(ptr_lower_limits, ptr_lower_limits + sz_lower_limits);

    auto* ptr_upper_limits = reinterpret_cast<double*>(taskData->inputs[3]);
    upper_limits.assign(ptr_upper_limits, ptr_upper_limits + sz_upper_limits);

    auto ptr_epsilon = reinterpret_cast<double*>(taskData->inputs[4]);
    epsilon = *ptr_epsilon;
  }
  return true;
}

bool kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  int size, ProcRank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  sz_t = get_mpi_type();
  if (ProcRank == 0) {
    // Check count elements of output
    return taskData->inputs_count[1] && taskData->inputs_count[2];
  }
  return true;
}

bool kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int size, ProcRank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

  MPI_Bcast(&sz_values, 1, sz_t, 0, MPI_COMM_WORLD);
  MPI_Bcast(&sz_lower_limits, 1, sz_t, 0, MPI_COMM_WORLD);
  MPI_Bcast(&sz_upper_limits, 1, sz_t, 0, MPI_COMM_WORLD);
  MPI_Bcast(&dim, 1, sz_t, 0, MPI_COMM_WORLD);
  MPI_Bcast(&epsilon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (ProcRank > 0) {
    f_values = std::vector<double>(sz_values);
    lower_limits = std::vector<double>(sz_lower_limits);
    upper_limits = std::vector<double>(sz_upper_limits);
  }
  MPI_Bcast(lower_limits.data(), sz_lower_limits, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(upper_limits.data(), sz_upper_limits, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(f_values.data(), static_cast<int>(sz_values), MPI_DOUBLE, 0, MPI_COMM_WORLD);  // problem?

  // std::cout << "Im process " << ProcRank << std::endl;
  //  std::cout << "my f_values is ";
  //  for (size_t i = 0; i < sz_values; i++) {
  //    std::cout << " " << f_values[i];
  //  }
  // std::cout << "my lower_limits is ";
  // for (size_t i = 0; i < sz_lower_limits; i++) {
  //   std::cout << " " << lower_limits[i];
  // }
  // std::cout << std::endl;
  // std::cout << "my upper_limits is ";
  // for (size_t i = 0; i < sz_upper_limits; i++) {
  //   std::cout << " " << upper_limits[i];
  // }
  // std::cout << std::endl;

  // if (ProcRank == 0) {
  //   std::cout << "mpi" << std::endl;
  // }
  run_multistep_scheme_method_rectangle(f, f_values, lower_limits, upper_limits, dim, epsilon);
  // if (ProcRank == 0) {
  //   std::cout << "global_result is " << I_2n;
  // }
  return true;
}

bool kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  int ProcRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  if (ProcRank == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = I_2n;
  }
  return true;  //
}

kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel::~TestMPITaskParallel() { MPI_Type_free(&sz_t); }
MPI_Datatype kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel::get_mpi_type() {
  MPI_Type_contiguous(sizeof(size_t), MPI_BYTE, &sz_t);
  MPI_Type_commit(&sz_t);
  return sz_t;
}
