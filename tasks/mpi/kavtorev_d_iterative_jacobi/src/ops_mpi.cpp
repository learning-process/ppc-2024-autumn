#include "mpi/kavtorev_d_iterative_jacobi/include/ops_mpi.hpp"

#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <cassert>
#include <cmath>

void calculate_sizes_displs(int N, int num_proc, std::vector<int>& sizes, std::vector<int>& displs) {
  sizes.resize(num_proc);
  displs.resize(num_proc);

  int base_size = N / num_proc;
  int remainder = N % num_proc;
  int offset = 0;

  for (int rank = 0; rank < num_proc; ++rank) {
    sizes[rank] = base_size + (rank < remainder ? 1 : 0);
    displs[rank] = offset;
    offset += sizes[rank];
  }
}

bool kavtorev_d_iterative_jacobi_mpi::IterativeJacobiParallelMPI::validation() {
  internal_order_test();

  if (world.rank() != 0) {
    return true;
  }

  if (!taskData) return false;

  if (taskData->inputs.size() < 5 || taskData->outputs.empty()) return false;

  if ((taskData->inputs[0] == nullptr) || (taskData->inputs[1] == nullptr) || (taskData->inputs[2] == nullptr) ||
      (taskData->inputs[3] == nullptr) || (taskData->inputs[4] == nullptr))
    return false;
  if (taskData->outputs[0] == nullptr) return false;

  int val_n = *reinterpret_cast<int*>(taskData->inputs[0]);
  if (val_n <= 0) return false;

  double val_eps = *reinterpret_cast<double*>(taskData->inputs[1]);
  if (val_eps <= 0.0) return false;

  int val_iterations = *reinterpret_cast<int*>(taskData->inputs[2]);
  if (val_iterations <= 0) return false;

  auto* val_A_data = reinterpret_cast<double*>(taskData->inputs[3]);
  int val_A_size = taskData->inputs_count[3];

  auto* val_F_data = reinterpret_cast<double*>(taskData->inputs[4]);
  int val_F_size = taskData->inputs_count[4];

  auto* val_output = reinterpret_cast<double*>(taskData->outputs[0]);
  int val_output_size = taskData->outputs_count[0];

  if ((val_A_data == nullptr) || (val_F_data == nullptr) || (val_output == nullptr)) return false;

  if (val_A_size != val_n * val_n) return false;

  if (val_F_size != val_n) return false;

  if (val_output_size < val_n) return false;

  std::vector<double> val_A_flat(val_A_data, val_A_data + val_A_size);
  std::vector<double> val_F(val_F_data, val_F_data + val_F_size);

  for (size_t i = 0; i < val_A_flat.size(); ++i)
    if (std::isnan(val_A_flat[i]) || std::isinf(val_A_flat[i])) return false;

  for (size_t i = 0; i < val_F.size(); ++i)
    if (std::isnan(val_F[i]) || std::isinf(val_F[i])) return false;

  for (int i = 0; i < val_n; ++i) {
    double diagonal_element = val_A_flat[i * val_n + i];

    if (diagonal_element == 0.0) return false;
  }

  for (int i = 0; i < val_n; ++i) {
    double sum = 0.0;
    for (int j = 0; j < val_n; ++j)
      if (j != i) sum += std::abs(val_A_flat[i * val_n + j]);

    double diagonal_element = std::abs(val_A_flat[i * val_n + i]);
    if (diagonal_element < sum) return false;
  }

  return true;
}

bool kavtorev_d_iterative_jacobi_mpi::IterativeJacobiParallelMPI::pre_processing() {
  internal_order_test();

  n = *reinterpret_cast<int*>(taskData->inputs[0]);
  eps = *reinterpret_cast<double*>(taskData->inputs[1]);
  iterations = *reinterpret_cast<int*>(taskData->inputs[2]);

  num_proc = world.size();
  rank = world.rank();

  if (rank == 0) {
    auto* A_data = reinterpret_cast<double*>(taskData->inputs[3]);
    int A_size = taskData->inputs_count[3];
    auto* F_data = reinterpret_cast<double*>(taskData->inputs[4]);
    int F_size = taskData->inputs_count[4];

    try {
      A_flat.assign(A_data, A_data + A_size);
      F.assign(F_data, F_data + F_size);
      result_vector.resize(n);
    } catch (const std::exception& e) {
      std::cerr << "Error during global data assignment: " << e.what() << std::endl;
      return false;
    }
  }

  calculate_sizes_displs(n, num_proc, sizes, displs);

  local_size = sizes[rank];
  local_displ = displs[rank];

  try {
    local_A_flat.resize(local_size * n);
    local_F.resize(local_size);
    X.resize(n, 0.0);
  } catch (const std::exception& e) {
    std::cerr << "Error during vector resizing: " << e.what() << std::endl;
    return false;
  }

  sendcounts_A.resize(num_proc);
  displs_A.resize(num_proc);
  for (int i = 0; i < num_proc; ++i) {
    sendcounts_A[i] = sizes[i] * n;
    displs_A[i] = displs[i] * n;
  }

  sendcounts_F = sizes;
  displs_F = displs;

  return true;
}

bool kavtorev_d_iterative_jacobi_mpi::IterativeJacobiParallelMPI::run() {
  internal_order_test();

  if (rank == 0) {
    boost::mpi::scatterv(world, A_flat.data(), sendcounts_A, displs_A, local_A_flat.data(), sendcounts_A[rank], 0);
    boost::mpi::scatterv(world, F.data(), sendcounts_F, displs_F, local_F.data(), sendcounts_F[rank], 0);
  } else {
    boost::mpi::scatterv(world, local_A_flat.data(), sendcounts_A[rank], 0);
    boost::mpi::scatterv(world, local_F.data(), sendcounts_F[rank], 0);
  }

  boost::mpi::broadcast(world, X, 0);

  std::vector<std::vector<double>> local_A(local_size, std::vector<double>(n));
  for (int i = 0; i < local_size; ++i) {
    for (int j = 0; j < n; ++j) {
      local_A[i][j] = local_A_flat[i * n + j];
    }
  }

  std::vector<double> TempX(n);
  double norm;

  int iteration = 0;
  do {
    std::vector<double> local_TempX(local_size);
    for (int i = 0; i < local_size; ++i) {
      int global_i = local_displ + i;
      double sum = local_F[i];
      for (int g = 0; g < n; ++g) {
        if (global_i != g) sum -= local_A[i][g] * X[g];
      }
      local_TempX[i] = sum / local_A[i][global_i];
    }

    if (rank == 0) {
      boost::mpi::gatherv(world, local_TempX.data(), local_TempX.size(), TempX.data(), sizes, displs, 0);
    } else {
      boost::mpi::gatherv(world, local_TempX.data(), local_TempX.size(), 0);
    }

    boost::mpi::broadcast(world, TempX, 0);

    double local_norm = 0.0;
    for (int i = 0; i < local_size; ++i) {
      int global_i = local_displ + i;
      double diff = fabs(X[global_i] - TempX[global_i]);
      if (diff > local_norm) local_norm = diff;
    }

    boost::mpi::all_reduce(world, local_norm, norm, boost::mpi::maximum<double>());

    X = TempX;

    iteration++;

  } while (iteration < iterations && norm > eps);

  return iteration != iterations;
}

bool kavtorev_d_iterative_jacobi_mpi::IterativeJacobiParallelMPI::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* mpi_X = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(X.begin(), X.end(), mpi_X);
  }

  return true;
}

bool kavtorev_d_iterative_jacobi_mpi::jacobi_method(int N, const std::vector<double>& A_flat,
                                                    const std::vector<double>& F, std::vector<double>& X, double eps,
                                                    int iterations) {
  std::vector<double> TempX(N);
  double norm;

  int iteration = 0;
  do {
    for (int i = 0; i < N; i++) {
      TempX[i] = F[i];
      for (int j = 0; j < N; j++) {
        if (i != j) TempX[i] -= A_flat[i * N + j] * X[j];
      }
      TempX[i] /= A_flat[i * N + i];
    }

    norm = std::fabs(X[0] - TempX[0]);
    for (int i = 0; i < N; i++) {
      double diff = std::fabs(X[i] - TempX[i]);
      if (diff > norm) norm = diff;
      X[i] = TempX[i];
    }
    iteration++;
  } while (iteration < iterations && norm > eps);
  return iteration != iterations;
}

bool kavtorev_d_iterative_jacobi_mpi::IterativeJacobiSequentialMPI::validation() {
  internal_order_test();

  if (!taskData) {
    return false;
  }

  if (taskData->inputs.size() < 5 || taskData->outputs.empty()) {
    return false;
  }

  if ((taskData->inputs[0] == nullptr) || (taskData->inputs[1] == nullptr) || (taskData->inputs[2] == nullptr) ||
      (taskData->inputs[3] == nullptr) || (taskData->inputs[4] == nullptr)) {
    return false;
  }

  if (taskData->outputs[0] == nullptr) {
    return false;
  }

  int val_n = *reinterpret_cast<int*>(taskData->inputs[0]);

  if (val_n <= 0) {
    return false;
  }

  double val_eps = *reinterpret_cast<double*>(taskData->inputs[1]);

  if (val_eps <= 0.0) {
    return false;
  }

  int val_iterations = *reinterpret_cast<int*>(taskData->inputs[2]);

  if (val_iterations <= 0) {
    return false;
  }

  auto* val_A_data = reinterpret_cast<double*>(taskData->inputs[3]);
  int val_A_size = taskData->inputs_count[3];

  auto* val_F_data = reinterpret_cast<double*>(taskData->inputs[4]);
  int val_F_size = taskData->inputs_count[4];

  auto* val_output = reinterpret_cast<double*>(taskData->outputs[0]);
  int val_output_size = taskData->outputs_count[0];

  if ((val_A_data == nullptr) || (val_F_data == nullptr) || (val_output == nullptr)) {
    return false;
  }

  if (val_A_size != val_n * val_n) {
    return false;
  }

  if (val_F_size != val_n) {
    return false;
  }

  if (val_output_size < val_n) {
    return false;
  }

  std::vector<double> val_A_flat(val_A_data, val_A_data + val_A_size);
  std::vector<double> val_F(val_F_data, val_F_data + val_F_size);

  for (size_t i = 0; i < val_A_flat.size(); ++i) {
    if (std::isnan(val_A_flat[i]) || std::isinf(val_A_flat[i])) {
      return false;
    }
  }

  for (size_t i = 0; i < val_F.size(); ++i) {
    if (std::isnan(val_F[i]) || std::isinf(val_F[i])) {
      return false;
    }
  }

  for (int i = 0; i < val_n; ++i) {
    double diagonal_element = val_A_flat[i * val_n + i];
    if (diagonal_element == 0.0) {
      return false;
    }
  }

  for (int i = 0; i < val_n; ++i) {
    double sum = 0.0;
    for (int j = 0; j < val_n; ++j) {
      if (j != i) {
        sum += std::abs(val_A_flat[i * val_n + j]);
      }
    }
    double diagonal_element = std::abs(val_A_flat[i * val_n + i]);
    if (diagonal_element < sum) {
      return false;
    }
  }

  return true;
}

bool kavtorev_d_iterative_jacobi_mpi::IterativeJacobiSequentialMPI::pre_processing() {
  internal_order_test();

  n = *reinterpret_cast<int*>(taskData->inputs[0]);
  eps = *reinterpret_cast<double*>(taskData->inputs[1]);
  iterations = *reinterpret_cast<int*>(taskData->inputs[2]);
  auto* A_data = reinterpret_cast<double*>(taskData->inputs[3]);
  int A_size = taskData->inputs_count[3];
  auto* F_data = reinterpret_cast<double*>(taskData->inputs[4]);
  int F_size = taskData->inputs_count[4];

  try {
    A_flat.assign(A_data, A_data + A_size);
    F.assign(F_data, F_data + F_size);
    result_vector.resize(n);
  } catch (const std::exception& e) {
    std::cerr << "Error during global data assignment: " << e.what() << std::endl;
    return false;
  }

  return true;
}

bool kavtorev_d_iterative_jacobi_mpi::IterativeJacobiSequentialMPI::run() {
  internal_order_test();

  return kavtorev_d_iterative_jacobi_mpi::jacobi_method(n, A_flat, F, result_vector, eps, iterations);
}

bool kavtorev_d_iterative_jacobi_mpi::IterativeJacobiSequentialMPI::post_processing() {
  internal_order_test();

  auto* output_data = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(result_vector.begin(), result_vector.end(), output_data);

  return true;
}
