#include "mpi/zolotareva_a_SLE_gradient_method/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <cmath>
#include <numeric>
#include <seq/zolotareva_a_SLE_gradient_method/include/ops_seq.hpp>
#include <vector>

bool zolotareva_a_SLE_gradient_method_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  if (static_cast<int>(taskData->inputs_count[0]) < 0 || static_cast<int>(taskData->inputs_count[1]) < 0 ||
      static_cast<int>(taskData->outputs_count[0]) < 0)
    return false;
  if (taskData->inputs_count.size() < 2 || taskData->inputs.size() < 2 || taskData->outputs.empty()) return false;

  if (static_cast<int>(taskData->inputs_count[0]) !=
      (static_cast<int>(taskData->inputs_count[1]) * static_cast<int>(taskData->inputs_count[1])))
    return false;

  if (taskData->outputs_count[0] != taskData->inputs_count[1]) return false;

  // проверка симметрии и положительной определённости
  const auto* A = reinterpret_cast<const double*>(taskData->inputs[0]);

  return is_positive_and_simm(A, static_cast<int>(taskData->inputs_count[1]));
}
bool zolotareva_a_SLE_gradient_method_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  n_ = static_cast<int>(taskData->inputs_count[1]);
  A_.resize(n_ * n_);
  b_.resize(n_);
  x_.resize(n_, 0.0);
  const auto* input_matrix = reinterpret_cast<const double*>(taskData->inputs[0]);
  const auto* input_vector = reinterpret_cast<const double*>(taskData->inputs[1]);

  for (int i = 0; i < n_; ++i) {
    b_[i] = input_vector[i];
    for (int j = 0; j < n_; ++j) {
      A_[i * n_ + j] = input_matrix[i * n_ + j];
    }
  }

  return true;
}

bool zolotareva_a_SLE_gradient_method_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  conjugate_gradient(A_, b_, x_, n_);
  return true;
}

bool zolotareva_a_SLE_gradient_method_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  auto* output_raw = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(x_.begin(), x_.end(), output_raw);
  return true;
}

bool zolotareva_a_SLE_gradient_method_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (static_cast<int>(taskData->inputs_count[0]) < 0 || static_cast<int>(taskData->inputs_count[1]) < 0 ||
        static_cast<int>(taskData->outputs_count[0]) < 0)
      return false;
    if (taskData->inputs_count.size() < 2 || taskData->inputs.size() < 2 || taskData->outputs.empty()) return false;

    if (static_cast<int>(taskData->inputs_count[0]) !=
        (static_cast<int>(taskData->inputs_count[1]) * static_cast<int>(taskData->inputs_count[1])))
      return false;

    if (taskData->outputs_count[0] != taskData->inputs_count[1]) return false;

    // проверка симметрии и положительной определённости
    const auto* A = reinterpret_cast<const double*>(taskData->inputs[0]);

    return zolotareva_a_SLE_gradient_method_mpi::TestMPITaskSequential::is_positive_and_simm(
        A, static_cast<int>(taskData->inputs_count[1]));
  }
  return true;
}

bool zolotareva_a_SLE_gradient_method_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    n_ = static_cast<int>(taskData->inputs_count[1]);
    const auto* input_matrix = reinterpret_cast<const double*>(taskData->inputs[0]);
    const auto* input_vector = reinterpret_cast<const double*>(taskData->inputs[1]);
    A_.resize(n_ * n_);
    b_.resize(n_);
    for (int i = 0; i < n_; ++i) {
      b_[i] = input_vector[i];
      for (int j = 0; j < n_; ++j) {
        A_[i * n_ + j] = input_matrix[i * n_ + j];
      }
    }
  }

  return true;
}

bool zolotareva_a_SLE_gradient_method_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int world_size = world.size();
  int rank = world.rank();
  boost::mpi::broadcast(world, n_, 0);

  int base_rows = n_ / world_size;
  int remainder = n_ % world_size;
  local_rows = base_rows;

  if (rank == 0) {
    local_rows += remainder;

    int start_row = local_rows;
    for (int proc = 1; proc < world_size; ++proc) {
      world.send(proc, 0, A_.data() + start_row * n_, base_rows * n_);
      world.send(proc, 1, b_.data() + start_row, base_rows);
      start_row += base_rows;
    }

    local_A_.resize(local_rows * n_);
    local_b_.resize(local_rows);
    std::copy(A_.begin(), A_.begin() + local_rows * n_, local_A_.begin());
    std::copy(b_.begin(), b_.begin() + local_rows, local_b_.begin());
  } else {
    local_A_.resize(local_rows * n_);
    local_b_.resize(local_rows);
    world.recv(0, 0, local_A_.data(), local_rows * n_);
    world.recv(0, 1, local_b_.data(), local_rows);
  }

  x_.assign(local_rows, 0.0);
  std::vector<double> r(local_b_);
  std::vector<double> p(r);

  int local_rows_0 = base_rows + remainder;
  std::vector<int> recvcounts(world_size, base_rows);
  recvcounts[0] = local_rows_0;
  std::vector<int> displs(world_size, 0);
  for (int i = 1; i < world_size; ++i) {
    displs[i] = displs[i - 1] + recvcounts[i - 1];
  }

  std::vector<double> global_p(n_);
  std::vector<double> Ap(local_rows);

  double rs_old = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);
  double rs_global_old;
  boost::mpi::all_reduce(world, rs_old, rs_global_old, std::plus<>());
  double initial_res_norm = std::sqrt(rs_global_old);
  double threshold = (initial_res_norm == 0.0) ? 1e-12 : (1e-12 * initial_res_norm);

  for (int iter = 0; iter <= n_; ++iter) {
    MPI_Allgatherv(p.data(), local_rows, MPI_DOUBLE, global_p.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                   world);

    for (int i = 0; i < local_rows; ++i) {
      Ap[i] = std::inner_product(&local_A_[i * n_], &local_A_[i * n_] + n_, global_p.begin(), 0.0);
    }

    double local_dot_pAp = std::inner_product(p.begin(), p.end(), Ap.begin(), 0.0);
    double global_dot_pAp;
    boost::mpi::all_reduce(world, local_dot_pAp, global_dot_pAp, std::plus<>());

    if (global_dot_pAp == 0.0) break;
    double alpha = rs_global_old / global_dot_pAp;

    for (int i = 0; i < local_rows; ++i) {
      x_[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }

    double local_rs_new = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);
    double rs_global_new;
    boost::mpi::all_reduce(world, local_rs_new, rs_global_new, std::plus<>());

    if (rs_global_new < threshold) break;
    double beta = rs_global_new / rs_global_old;
    for (int i = 0; i < local_rows; ++i) p[i] = r[i] + beta * p[i];
    rs_global_old = rs_global_new;
  }

  if (world.rank() == 0) {
    X_.resize(n_);
    std::copy(x_.begin(), x_.end(), X_.begin());
    int start_row = local_rows;

    std::vector<double> buffer(base_rows);
    for (int proc = 1; proc < world.size(); ++proc) {
      world.recv(proc, 2, buffer);
      std::copy(buffer.begin(), buffer.end(), X_.begin() + start_row);
      start_row += base_rows;
    }
  } else
    world.send(0, 2, x_);

  return true;
}

bool zolotareva_a_SLE_gradient_method_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* output_raw = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(X_.begin(), X_.end(), output_raw);
  }

  return true;
}

void zolotareva_a_SLE_gradient_method_mpi::TestMPITaskSequential::conjugate_gradient(const std::vector<double>& A,
                                                                                     const std::vector<double>& b,
                                                                                     std::vector<double>& x, int N) {
  double initial_res_norm = 0.0;
  dot_product(initial_res_norm, b, b, N);
  initial_res_norm = std::sqrt(initial_res_norm);
  double threshold = initial_res_norm == 0.0 ? 1e-12 : (1e-12 * initial_res_norm);

  std::vector<double> r = b;  // начальный вектор невязки r = b - A*x0, x0 = 0
  std::vector<double> p = r;  // начальное направление поиска p = r
  double rs_old = 0;
  dot_product(rs_old, r, r, N);

  for (int s = 0; s <= N; ++s) {
    std::vector<double> Ap(N, 0.0);
    matrix_vector_mult(A, p, Ap, N);
    double pAp = 0.0;
    dot_product(pAp, p, Ap, N);
    if (pAp == 0.0) break;

    double alpha = rs_old / pAp;

    for (int i = 0; i < N; ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }

    double rs_new = 0.0;
    dot_product(rs_new, r, r, N);
    if (rs_new < threshold) {  // Проверка на сходимость
      break;
    }
    double beta = rs_new / rs_old;
    for (int i = 0; i < N; ++i) {
      p[i] = r[i] + beta * p[i];
    }

    rs_old = rs_new;
  }
}

void zolotareva_a_SLE_gradient_method_mpi::TestMPITaskSequential::dot_product(double& sum,
                                                                              const std::vector<double>& vec1,
                                                                              const std::vector<double>& vec2, int n) {
  for (int i = 0; i < n; ++i) {
    sum += vec1[i] * vec2[i];
  }
}

void zolotareva_a_SLE_gradient_method_mpi::TestMPITaskSequential::matrix_vector_mult(const std::vector<double>& matrix,
                                                                                     const std::vector<double>& vector,
                                                                                     std::vector<double>& result,
                                                                                     int n) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      result[i] += matrix[i * n + j] * vector[j];
    }
  }
}

bool zolotareva_a_SLE_gradient_method_mpi::TestMPITaskSequential::is_positive_and_simm(const double* A, int n) {
  std::vector<double> M(n * n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double val = A[i * n + j];
      M[i * n + j] = val;
      if (j > i) {
        if (val != A[j * n + i]) {
          return false;
        }
      }
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      double sum = M[i * n + j];
      for (int k = 0; k < j; k++) {
        sum -= M[i * n + k] * M[j * n + k];
      }
      if (i == j) {
        if (sum <= 1e-15) {
          return false;
        }
        M[i * n + j] = std::sqrt(sum);
      } else {
        M[i * n + j] = sum / M[j * n + j];
      }
    }
  }
  return true;
}