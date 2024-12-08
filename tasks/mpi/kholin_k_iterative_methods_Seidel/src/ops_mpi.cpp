#include "mpi/kholin_k_iterative_methods_Seidel/include/ops_mpi.hpp"

#include <memory.h>

#include <algorithm>
#include <functional>
#include <random>
using namespace std::chrono_literals;

bool kholin_k_iterative_methods_Seidel_mpi::IsDiagPred(std::vector<float> row_coeffs, size_t num_colls,
                                                       size_t start_index, size_t index) {
  float diag_element = std::fabs(row_coeffs[index]);
  float abs_sum = 0;
  float abs_el = 0;
  size_t size = num_colls;
  for (size_t j = start_index; j < start_index + size; j++) {
    if (j == index) {
      continue;
    }
    abs_el = std::fabs(row_coeffs[j]);
    abs_sum += abs_el;
  }
  return diag_element > abs_sum;
}

float kholin_k_iterative_methods_Seidel_mpi::gen_float_value() {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<float> coeff(1, 100);

  return coeff(gen);
}

std::vector<float> kholin_k_iterative_methods_Seidel_mpi::gen_matrix_with_diag_pred(size_t num_rows, size_t num_colls,
                                                                                    float p1, float p2) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<float> A_(num_rows * num_colls, 0.0f);
  // float mult = 100 * 100;
  std::uniform_real_distribution<float> coeff_diag(p1, p2);
  std::uniform_real_distribution<float> coeff_no_diag(-10, 100);

  for (size_t i = 0; i < num_rows; i++) {
    do {
      for (size_t j = 0; j < num_colls; j++) {
        if (i == j) {
          A_[num_colls * i + j] = coeff_diag(gen);
        } else {
          A_[num_colls * i + j] = coeff_no_diag(gen);
        }
      }
    } while (!IsDiagPred(A_, num_colls, num_colls * i, num_colls * i + i));
  }
  return A_;
}

void kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential::SetDefault() {
  X_next = std::vector<float>(n_rows, 0.0f);
  X_prev = std::vector<float>(n_rows, 0.0f);
  X = std::vector<float>(n_rows, 1.0f);
  B = std::vector<float>(n_rows);
  X0 = std::vector<float>(n_rows);
}

int kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential::rank(std::vector<float> local_matrix, size_t n,
                                                                       size_t m) {
  int rank = 0;

  for (size_t i = 0; i < std::min(n, m); ++i) {
    size_t max_row = i;
    for (size_t k = i + 1; k < n; ++k) {
      if (std::abs(local_matrix[k * m + i]) > std::abs(local_matrix[max_row * m + i])) {
        max_row = k;
      }
    }

    if (std::abs(local_matrix[max_row * m + i]) < std::numeric_limits<double>::epsilon()) {
      continue;
    }
    std::swap(local_matrix[i], local_matrix[max_row]);

    for (size_t j = i + 1; j < n; ++j) {
      double factor = local_matrix[j * m + i] / local_matrix[i * m + i];
      for (size_t k = i; k < m; ++k) {
        local_matrix[j * m + k] -= factor * local_matrix[i * m + k];
      }
    }
    rank++;
  }
  return rank;
}

bool kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  n_rows = taskData->inputs_count[0];
  n_colls = taskData->inputs_count[1];

  SetDefault();

  auto* ptr = reinterpret_cast<float*>(taskData->inputs[1]);
  epsilon = *ptr;

  auto* ptr_vector_X0 = reinterpret_cast<float*>(taskData->inputs[2]);
  std::memcpy(X0.data(), ptr_vector_X0, sizeof(float) * n_rows);

  iteration_perfomance();
  return true;
}

bool kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  auto* matrix = reinterpret_cast<float*>(taskData->inputs[0]);
  auto* B_ = reinterpret_cast<float*>(taskData->inputs[3]);

  size_t num_rows = taskData->inputs_count[0];
  size_t num_colls = taskData->inputs_count[1];
  if (!IsQuadro(num_rows, num_colls)) {
    return false;
  }
  A.assign(matrix, matrix + (num_rows * num_colls));
  if (!CheckDiagPred(matrix, num_rows, num_colls)) {
    return false;
  }
  std::vector<float> matrix_extended(num_rows * num_colls + 1);
  B.assign(B_, B_ + num_rows);
  size_t k = 0;
  for (size_t i = 0; i < num_rows; i++) {
    for (size_t j = 0; j < num_colls; j++) {
      matrix_extended[num_colls + 1 * i + j] = A[num_colls * i + j];
      if (j + 1 == num_colls) {
        k = j + 1;
      }
    }
    matrix_extended[num_colls + 1 * i + k] = B[i];
  }
  int rank_A = rank(A, num_rows, num_colls);
  int rank_A_ = rank(matrix_extended, num_rows, num_colls);
  bool IsSingleDecision = rank_A == rank_A_;
  if (!IsSingleDecision) {
    return IsSingleDecision;
  }
  return true;
}

bool kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  if (op == list_ops::METHOD_SEIDEL) {
    method_Seidel();
  }
  return true;
}

bool kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  auto* ptr = reinterpret_cast<float*>(taskData->outputs[0]);
  std::copy(X.begin(), X.end(), ptr);
  return true;
}

bool kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential::IsQuadro(const size_t num_rows,
                                                                            const size_t num_colls) {
  return num_rows == num_colls;
}

bool kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential::CheckDiagPred(float matrix[], const size_t num_rows,
                                                                                 const size_t num_colls) {
  size_t rows = num_rows;
  size_t colls = num_colls;
  float abs_diag_element = 0.0f;
  float abs_el = 0.0f;
  float abs_sum = 0.0f;
  for (size_t i = 0; i < rows; i++) {
    abs_diag_element = std::fabs(matrix[colls * i + i]);
    for (size_t j = 0; j < colls; j++) {
      if (j == i) {
        continue;
      }
      abs_el = fabsf(matrix[colls * i + j]);
      abs_sum += abs_el;
    }
    if (abs_diag_element <= abs_sum) {
      return false;
    }
    abs_sum = 0;
  }
  return true;
}

std::vector<float> kholin_k_iterative_methods_Seidel_mpi::gen_vector(size_t sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<float> row(sz);
  std::uniform_real_distribution<float> coeff(-100, 100);
  for (size_t i = 0; i < sz; i++) {
    row[i] = coeff(gen);
  }
  return row;
}

void kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential::iteration_perfomance() {
  C = std::vector<float>(n_rows * n_colls, 0.0f);
  for (size_t i = 0; i < n_rows; i++) {
    for (size_t j = 0; j < n_colls; j++) {
      if (i == j) {
        B[i] = B[i] / A[n_colls * i + i];
        C[n_colls * i + i] = 0;
        continue;
      }
      C[n_colls * i + j] = -A[n_colls * i + j] / A[n_colls * i + i];
    }
  }
  std::copy(B.begin(), B.end(), X0.begin());
}

float kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential::d() {
  // AX + B < epsilon
  float d = 0;
  float maxd = 0;
  for (size_t i = 0; i < n_rows; i++) {
    d = std::fabs(X_next[i] - X_prev[i]);
    if (d > maxd) {
      maxd = d;
    }
  }
  return maxd;
}

void kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential::method_Seidel() {
  float delta = 1.0f;
  for (size_t k_iteration = 0; delta > epsilon; k_iteration++) {
    for (size_t i = 0; i < n_rows; i++) {
      for (size_t j = 0; j < n_colls; j++) {
        if (j < i) {
          X_next[i] += C[n_colls * i + j] * X_next[j];
        } else if (j > i) {
          if (k_iteration == 0) {
            X_next[i] += C[n_colls * i + j] * X0[j];
          } else {
            X_next[i] += C[n_colls * i + j] * X_prev[j];
          }
        }
      }
      X_next[i] += B[i];
    }
    delta = d();
    std::copy(X_next.begin(), X_next.end(), X_prev.begin());
    std::fill(X_next.begin(), X_next.end(), 0.0f);
  }
  std::copy(X_prev.begin(), X_prev.end(), X.begin());
}

void kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel::SetDefault() {
  max_delta = 100000.0f;
  global_x = 0.0f;
  count = 0.0f;
  X = std::vector<float>(n_rows, 1.0f);
  X0 = std::vector<float>(n_rows);
  B = std::vector<float>(n_rows);
  X_next = std::vector<float>(n_rows, 0.0f);
  X_prev = std::vector<float>(n_rows, 0.0f);
}

bool kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  int ProcRank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (ProcRank == 0) {
    n_rows = taskData->inputs_count[0];
    n_colls = taskData->inputs_count[1];
    SetDefault();
  }
  if (ProcRank == 0) {
    auto* ptr = reinterpret_cast<float*>(taskData->inputs[1]);
    epsilon = *ptr;

    auto* ptr_vector_X0 = reinterpret_cast<float*>(taskData->inputs[2]);
    std::memcpy(X0.data(), ptr_vector_X0, sizeof(float) * n_rows);

    count = ((n_rows * n_colls) - n_rows) / 2;
  }
  return true;
}

int kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel::rank(std::vector<float> local_matrix, size_t n,
                                                                     size_t m) {
  int rank = 0;

  for (size_t i = 0; i < std::min(n, m); ++i) {
    size_t max_row = i;
    for (size_t k = i + 1; k < n; ++k) {
      if (std::abs(local_matrix[k * m + i]) > std::abs(local_matrix[max_row * m + i])) {
        max_row = k;
      }
    }

    if (std::abs(local_matrix[max_row * m + i]) < std::numeric_limits<double>::epsilon()) {
      continue;
    }
    std::swap(local_matrix[i], local_matrix[max_row]);

    for (size_t j = i + 1; j < n; ++j) {
      double factor = local_matrix[j * m + i] / local_matrix[i * m + i];
      for (size_t k = i; k < m; ++k) {
        local_matrix[j * m + k] -= factor * local_matrix[i * m + k];
      }
    }
    rank++;
  }
  return rank;
}

bool kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  int ProcRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  sz_t = get_mpi_type();
  if (ProcRank == 0) {
    auto* matrix = reinterpret_cast<float*>(taskData->inputs[0]);
    auto* B_ = reinterpret_cast<float*>(taskData->inputs[3]);

    size_t num_rows = taskData->inputs_count[0];
    size_t num_colls = taskData->inputs_count[1];
    if (!IsQuadro(num_rows, num_colls)) {
      return false;
    }
    A.assign(matrix, matrix + (num_rows * num_colls));
    if (!CheckDiagPred(matrix, num_rows, num_colls)) {
      return false;
    }
    std::vector<float> matrix_extended(num_rows * num_colls + 1);
    B.assign(B_, B_ + num_rows);
    size_t k = 0;
    for (size_t i = 0; i < num_rows; i++) {
      for (size_t j = 0; j < num_colls; j++) {
        matrix_extended[num_colls + 1 * i + j] = A[num_colls * i + j];
        if (j + 1 == num_colls) {
          k = j + 1;
        }
      }
      matrix_extended[num_colls + 1 * i + k] = B[i];
    }
    int rank_A = rank(A, num_rows, num_colls);
    int rank_A_ = rank(matrix_extended, num_rows, num_colls);
    bool IsSingleDecision = rank_A == rank_A_;
    if (!IsSingleDecision) {
      return IsSingleDecision;
    }
  }
  return true;
}

bool kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int ProcRank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Bcast(&n_rows, 1, sz_t, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n_colls, 1, sz_t, 0, MPI_COMM_WORLD);
  MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);
  upper_send_counts = std::vector<float>(n_rows * size);
  lower_send_counts = std::vector<float>(n_rows * size);
  upper_C = std::vector<float>(count);
  lower_C = std::vector<float>(count);
  local_upper_counts = std::vector<float>(n_rows);
  local_lower_counts = std::vector<float>(n_rows);
  upper_displs = std::vector<float>(n_rows * size);
  lower_displs = std::vector<float>(n_rows * size);
  if (ProcRank > 0) {
    B = std::vector<float>(n_rows);
    X_next = std::vector<float>(n_rows);
    X_prev = std::vector<float>(n_rows);
    X0 = std::vector<float>(n_rows);
  }
  MPI_Bcast(X_next.data(), n_rows, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(X_prev.data(), n_rows, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&max_delta, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  if (ProcRank == 0) {
    iteration_perfomance();
  }
  if (op == list_ops::METHOD_SEIDEL) {
    method_Seidel();
  }
  return true;
}

bool kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  int ProcRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  if (ProcRank == 0) {
    auto* ptr = reinterpret_cast<float*>(taskData->outputs[0]);
    std::copy(X.begin(), X.end(), ptr);
  }
  return true;
}

bool kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel::IsQuadro(size_t num_rows, size_t num_colls) {
  return num_rows == num_colls;
}

bool kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel::CheckDiagPred(float matrix[], size_t num_rows,
                                                                               size_t num_colls) {
  size_t rows = num_rows;
  size_t colls = num_colls;
  float abs_diag_element = 0.0f;
  float abs_el = 0.0f;
  float abs_sum = 0.0f;
  for (size_t i = 0; i < rows; i++) {
    abs_diag_element = fabsf(matrix[colls * i + i]);
    for (size_t j = 0; j < colls; j++) {
      if (j == i) {
        continue;
      }
      abs_el = fabsf(matrix[colls * i + j]);
      abs_sum += abs_el;
    }
    if (abs_diag_element <= abs_sum) {
      return false;
    }
    abs_sum = 0;
  }
  return true;
}

std::vector<float> kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel::gen_vector(size_t sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<float> row = std::vector<float>(sz);
  std::uniform_real_distribution<float> coeff(10, 1000);
  for (size_t i = 0; i < sz; i++) {
    row[i] = coeff(gen);
  }
  return row;
}

void kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel::iteration_perfomance() {
  int ProcRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  C = std::vector<float>(n_rows * n_colls, 0.0f);
  for (size_t i = 0; i < n_rows; i++) {
    for (size_t j = 0; j < n_colls; j++) {
      if (i == j) {
        B[i] = B[i] / A[n_colls * i + i];
        C[n_colls * i + i] = 0;
        continue;
      }
      C[n_colls * i + j] = -A[n_colls * i + j] / A[n_colls * i + i];
    }
  }
  std::copy(B.begin(), B.end(), X0.begin());
}

float kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel::d() {
  // AX+B < epsilon
  float d = 0.0f;
  float maxd = 0.0f;
  int ProcRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  int start_row = upper_displs[ProcRank] - 1;
  for (int i = upper_displs[ProcRank] - 1, j = ProcRank; i < start_row + upper_send_counts[j] + 1; i++) {
    d = std::fabs(X_next[i] - X_prev[i]);
    if (d > maxd) {
      maxd = d;
    }
  }
  return maxd;
}

void kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel::method_Seidel() {
  int ProcRank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  float delta = 1000000.0f;
  to_upper_diag_matrix();
  to_lower_diag_matrix();
  MPI_Bcast(&epsilon, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(X0.data(), n_rows, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(B.data(), n_rows, MPI_FLOAT, 0, MPI_COMM_WORLD);
  float local_x = 0.0f;
  int start_row = 0.0f;
  for (size_t k_iteration = 0; max_delta > epsilon; k_iteration++) {
    for (size_t i = 0; i < n_rows; i++) {
      start_row = lower_displs[ProcRank + i * size] - n_colls * i;
      for (int l = lower_displs[ProcRank + i * size] - n_colls * i, j = l; l < start_row + local_lower_counts[i];
           l++, j++) {
        local_x += lower_C[l] * X_next[j];
      }
      if (k_iteration == 0) {
        start_row = upper_displs[ProcRank + i * size] - n_colls * i;
        for (int u = upper_displs[ProcRank + i * size] - n_colls * i, j = u; u < start_row + local_upper_counts[i];
             u++, j++) {
          local_x += upper_C[u] * X0[j];
        }
      } else {
        start_row = upper_displs[ProcRank + i * size] - n_colls * i;
        for (int u = upper_displs[ProcRank + i * size] - n_colls * i, j = u; u < start_row + local_upper_counts[i];
             u++, j++) {
          local_x += upper_C[u] * X_prev[j];
        }
      }
      MPI_Allreduce(&local_x, &global_x, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      X_next[i] = global_x + B[i];
      local_x = 0.0f;
    }
    delta = d();
    MPI_Allreduce(&delta, &max_delta, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    std::copy(X_next.begin(), X_next.end(), X_prev.begin());
    std::fill(X_next.begin(), X_next.end(), 0.0f);
  }
  if (ProcRank == 0) {
    std::copy(X_prev.begin(), X_prev.end(), X.begin());
  }
}

void kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel::to_upper_diag_matrix() {
  int ProcRank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int size_portion = 0;
  int residue = 0;
  int start_row = 0;
  int end_row = 0;
  float coeff = 0.0f;
  int counter = 0;
  if (ProcRank == 0) {
    for (size_t i = 0; i < n_rows; i++) {
      size_portion = (n_colls - 1 - i) / size;
      if (size_portion == 0) {
        for (int p = 0; p < size; p++) {
          if (p == 0) {
            upper_send_counts[p + i * size] = n_colls - 1 - i;
            upper_displs[p + i * size] = i * n_colls + i + 1;

            start_row = upper_displs[p + i * size];
            end_row = upper_send_counts[p + i * size];
            for (int k = start_row; k < start_row + end_row; k++) {
              coeff = C[k];
              upper_C[counter] = coeff;
              counter++;
            }
          } else {
            upper_send_counts[p + i * size] = 0;
            upper_displs[p + i * size] = 0;
          }
        }
      } else {
        residue = (n_colls - 1 - i) % size;
        for (int p = 0; p < size; p++) {
          upper_send_counts[p + i * size] = size_portion + (p < residue ? 1 : 0);
          upper_displs[p + i * size] =
              (p == 0 ? (i * n_colls + i + 1)
                      : upper_displs[(p + i * size) - 1] + upper_send_counts[(p + i * size) - 1]);

          start_row = upper_displs[p + i * size];
          end_row = upper_send_counts[p + i * size];
          for (int k = start_row; k < start_row + end_row; k++) {
            coeff = C[k];
            upper_C[counter] = coeff;
            counter++;
          }
        }
      }
    }
  }
  MPI_Bcast(upper_C.data(), count, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(upper_send_counts.data(), n_rows * size, MPI_INT, 0, MPI_COMM_WORLD);
  for (size_t i = 0; i < n_rows; i++) {
    local_upper_counts[i] = upper_send_counts[ProcRank + size * i];
  }
  MPI_Bcast(upper_displs.data(), n_rows * size, MPI_INT, 0, MPI_COMM_WORLD);
}
void kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel::to_lower_diag_matrix() {
  int ProcRank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int size_portion = 0;
  int residue = 0;
  int start_row = 0;
  int end_row = 0;
  float coeff = 0.0f;
  int counter = 0;
  if (ProcRank == 0) {
    for (size_t i = 0; i < n_rows; i++) {
      size_portion = (i) / size;
      if (size_portion == 0) {
        for (int p = 0; p < size; p++) {
          if (p == 0) {
            lower_send_counts[p + i * size] = i;
            lower_displs[p + i * size] = i * n_colls;

            start_row = lower_displs[p + i * size];
            end_row = lower_send_counts[p + i * size];
            for (int k = start_row; k < start_row + end_row; k++) {
              coeff = C[k];
              lower_C[counter] = coeff;
              counter++;
            }
          } else {
            lower_send_counts[p + i * size] = 0;
            lower_displs[p + i * size] = 0;
          }
        }
      } else {
        residue = (i) % size;
        for (int p = 0; p < size; p++) {
          lower_send_counts[p + i * size] = size_portion + (p < residue ? 1 : 0);
          lower_displs[p + i * size] =
              (p == 0 ? (i * n_colls) : lower_displs[(p + i * size) - 1] + lower_send_counts[(p + i * size) - 1]);

          start_row = lower_displs[p + i * size];
          end_row = lower_send_counts[p + i * size];
          for (int k = start_row; k < start_row + end_row; k++) {
            coeff = C[k];
            lower_C[counter] = coeff;
            counter++;
          }
        }
      }
    }
  }
  MPI_Bcast(lower_C.data(), count, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(lower_send_counts.data(), n_rows * size, MPI_INT, 0, MPI_COMM_WORLD);
  for (size_t i = 0; i < n_rows; i++) {
    local_lower_counts[i] = lower_send_counts[ProcRank + size * i];
  }
  MPI_Bcast(lower_displs.data(), n_rows * size, MPI_INT, 0, MPI_COMM_WORLD);
}

kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel::~TestMPITaskParallel() { MPI_Type_free(&sz_t); }
MPI_Datatype kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel::get_mpi_type() {
  MPI_Type_contiguous(sizeof(size_t), MPI_BYTE, &sz_t);
  MPI_Type_commit(&sz_t);
  return sz_t;
}