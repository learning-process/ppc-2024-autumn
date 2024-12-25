#include "mpi/deryabin_m_jacobi_iterative_method/include/ops_mpi.hpp"

#include <thread>

bool deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskSequential::pre_processing() {
  internal_order_test();
  input_right_vector_ = std::vector<double>(taskData->inputs_count[1]);
  auto* tmp_ptr_vec = reinterpret_cast<double*>(taskData->inputs[1]);
  std::copy(tmp_ptr_vec, tmp_ptr_vec + taskData->inputs_count[1], input_right_vector_.begin());
  output_x_vector_ = std::vector<double>(input_right_vector_.size());
  return true;
}

bool deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskSequential::validation() {
  internal_order_test();
  input_matrix_ = std::vector<double>(taskData->inputs_count[0]);
  auto* tmp_ptr_matr = reinterpret_cast<double*>(taskData->inputs[0]);
  std::copy(tmp_ptr_matr, tmp_ptr_matr + taskData->inputs_count[0], input_matrix_.begin());
  unsigned short i = 0;
  unsigned short j;
  unsigned short n = taskData->inputs_count[1];
  while (i != n) {
    if (i == 0) {
      j = 0;
      while (j != n) {
        if (input_matrix_[j] < 0) {
          return false;
        }
        j++;
      }
      if (input_matrix_[0] <= std::accumulate(input_matrix_.begin() + 1, input_matrix_.begin() + n - 1, 0)) {
        return false;
      }
    } else {
      j = 0;
      while (j != n) {
        if (input_matrix_[i * n + j] < 0) {
          return false;
        }
        j++;
      }
      if (i > 0 && i < n - 1) {
        if (input_matrix_[i * (n + 1)] <=
            std::accumulate(input_matrix_.begin() + i * n, input_matrix_.begin() + i * (n + 1) - 1, 0) +
                std::accumulate(input_matrix_.begin() + i * (n + 1) + 1, input_matrix_.begin() + (i + 1) * n - 1, 0)) {
          return false;
        }
      }
      if (i == n - 1) {
        if (input_matrix_[i * (n + 1)] <= std::accumulate(input_matrix_.begin() + i * n, input_matrix_.end() - 1, 0)) {
          return false;
        }
      }
    }
    i++;
  }
  return taskData->outputs_count[0] == 1;
}

bool deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskSequential::run() {
  internal_order_test();
  unsigned short Nmax = 10000;
  unsigned short num_of_iterations = 0;
  double epsilon = pow(10, -6);
  double max_delta_x_i = 0;
  std::vector<double> x_old;
  unsigned short n;
  do {
    x_old = output_x_vector_;
    unsigned short i = 0;
    unsigned short j;
    n = taskData->inputs_count[1];
    double sum;
    while (i != n) {
      j = 0;
      sum = 0;
      while (j != n) {
        if (i != j) {
          sum += input_matrix_[i * n + j] * x_old[j];
        }
        j++;
      }
      output_x_vector_[i] = (input_right_vector_[i] - sum) * (1.0 / input_matrix_[i * (n + 1)]);
      if (std::abs(output_x_vector_[i] - x_old[i]) > max_delta_x_i) {
        max_delta_x_i = std::abs(output_x_vector_[i] - x_old[i]);
      }
      i++;
    }
    num_of_iterations++;
  } while (num_of_iterations < Nmax && max_delta_x_i > epsilon);
  return true;
}

bool deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<std::vector<double>*>(taskData->outputs[0])[0] = output_x_vector_;
  return true;
}

bool deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input_right_vector_ = std::vector<double>(taskData->inputs_count[0]);
    auto* tmp_ptr_vec = reinterpret_cast<double*>(taskData->inputs[1]);
    std::copy(tmp_ptr_vec, tmp_ptr_vec + taskData->inputs_count[0], input_right_vector_.begin());
  }
  return true;
}

bool deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    unsigned short n = taskData->inputs_count[0];
    input_matrix_ = std::vector<double>(n * n);
    auto* tmp_ptr_matr = reinterpret_cast<double*>(taskData->inputs[0]);
    std::copy(tmp_ptr_matr, tmp_ptr_matr + n * n, input_matrix_.begin());
    unsigned short i = 0;
    unsigned short j;
    while (i != n) {
      if (i == 0) {
        j = 0;
        while (j != n) {
          if (input_matrix_[j] < 0) {
            return false;
          }
          j++;
        }
        if (input_matrix_[0] <= std::accumulate(input_matrix_.begin() + 1, input_matrix_.begin() + n - 1, 0)) {
          return false;
        }
      } else {
        j = 0;
        while (j != n) {
          if (input_matrix_[i * n + j] < 0) {
            return false;
          }
          j++;
        }
        if (i > 0 && i < n - 1) {
          if (input_matrix_[i * (n + 1)] <=
              std::accumulate(input_matrix_.begin() + i * n, input_matrix_.begin() + i * (n + 1) - 1, 0) +
                  std::accumulate(input_matrix_.begin() + i * (n + 1) + 1, input_matrix_.begin() + (i + 1) * n - 1,
                                  0)) {
            return false;
          }
        }
        if (i == n - 1) {
          if (input_matrix_[i * (n + 1)] <=
              std::accumulate(input_matrix_.begin() + i * n, input_matrix_.end() - 1, 0)) {
            return false;
          }
        }
      }
      i++;
    }
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskParallel::run() {
  internal_order_test();
  unsigned short Nmax = 10000;
  unsigned short num_of_iterations = 0;
  double epsilon = pow(10, -6);
  double max_delta_x_i = 0;
  std::vector<double> x_old;
  unsigned short n;
  unsigned short i;
  unsigned short j;
  double sum;
  if (world.size() == 1) {
    n = taskData->inputs_count[0];
    output_x_vector_ = std::vector<double>(n);
    do {
      x_old = output_x_vector_;
      i = 0;
      while (i != n) {
        j = 0;
        sum = 0;
        while (j != n) {
          if (i != j) {
            sum += input_matrix_[i * n + j] * x_old[j];
          }
          j++;
        }
        output_x_vector_[i] = (input_right_vector_[i] - sum) * (1.0 / input_matrix_[i * (n + 1)]);
        if (std::abs(output_x_vector_[i] - x_old[i]) > max_delta_x_i) {
          max_delta_x_i = std::abs(output_x_vector_[i] - x_old[i]);
        }
        i++;
      }
      num_of_iterations++;
    } while (num_of_iterations < Nmax && max_delta_x_i > epsilon);
    return true;
  }
  unsigned short number_of_local_matrix_rows = 0;
  unsigned short ostatochnoe_chislo_strock = 0;
  n = 0;
  std::vector<int> displacements(world.size());
  if (world.rank() == 0) {
    n = taskData->inputs_count[0];
    number_of_local_matrix_rows = n / world.size();
    ostatochnoe_chislo_strock = n % world.size();
    for (int proc = 1; proc < world.size(); proc++) {
      displacements[proc] = number_of_local_matrix_rows * (proc - 1);
    }
  }
  boost::mpi::broadcast(world, number_of_local_matrix_rows, 0);
  boost::mpi::broadcast(world, displacements.data(), displacements.size(), 0);
  boost::mpi::broadcast(world, n, 0);
  output_x_vector_ = std::vector<double>(n);
  local_input_matrix_part_ = std::vector<double>(number_of_local_matrix_rows * n);
  local_input_right_vector_part_ = std::vector<double>(number_of_local_matrix_rows);
  std::vector<int> sendcounts(world.size(), number_of_local_matrix_rows);
  if (world.rank() == 0) {
    sendcounts[world.rank()] = number_of_local_matrix_rows + ostatochnoe_chislo_strock;
    displacements[world.rank()] = n - number_of_local_matrix_rows - ostatochnoe_chislo_strock;
    local_input_matrix_part_ = std::vector<double>((number_of_local_matrix_rows + ostatochnoe_chislo_strock) * n);
    std::copy(input_matrix_.begin() + n * (n - number_of_local_matrix_rows - ostatochnoe_chislo_strock),
              input_matrix_.begin() + n * n, local_input_matrix_part_.begin());
    local_input_right_vector_part_ = std::vector<double>(number_of_local_matrix_rows + ostatochnoe_chislo_strock);
    std::copy(input_right_vector_.begin() + n - number_of_local_matrix_rows - ostatochnoe_chislo_strock,
              input_right_vector_.begin() + n, local_input_right_vector_part_.begin());
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_matrix_.data() + (proc - 1) * number_of_local_matrix_rows * n,
                 number_of_local_matrix_rows * n);
      world.send(proc, 0, input_right_vector_.data() + (proc - 1) * number_of_local_matrix_rows,
                 number_of_local_matrix_rows);
    }
  } else {
    world.recv(0, 0, local_input_matrix_part_.data(), number_of_local_matrix_rows * n);
    world.recv(0, 0, local_input_right_vector_part_.data(), number_of_local_matrix_rows);
  }
  local_output_x_vector_part_ = std::vector<double>(local_input_right_vector_part_.size());
  do {
    x_old = output_x_vector_;
    i = 0;
    while (i != local_output_x_vector_part_.size()) {
      j = 0;
      sum = 0;
      if (world.rank() == 0) {
        while (j != n) {
          if (n - (number_of_local_matrix_rows + ostatochnoe_chislo_strock - i) != j) {
            sum += local_input_matrix_part_[i * n + j] * x_old[j];
          }
          j++;
        }
        local_output_x_vector_part_[i] =
            (local_input_right_vector_part_[i] - sum) *
            (1.0 /
             local_input_matrix_part_[(i + 1) * n - (number_of_local_matrix_rows + ostatochnoe_chislo_strock - i)]);
        if (std::abs(local_output_x_vector_part_[i] -
                     x_old[n - (number_of_local_matrix_rows + ostatochnoe_chislo_strock - i)]) > max_delta_x_i) {
          max_delta_x_i = std::abs(local_output_x_vector_part_[i] -
                                   x_old[n - (number_of_local_matrix_rows + ostatochnoe_chislo_strock - i)]);
        }
      } else {
        while (j != n) {
          if (i + (world.rank() - 1) * (number_of_local_matrix_rows) != j) {
            sum += local_input_matrix_part_[i * n + j] * x_old[j];
          }
          j++;
        }
        local_output_x_vector_part_[i] =
            (local_input_right_vector_part_[i] - sum) *
            (1.0 / local_input_matrix_part_[i * (n + 1) + (world.rank() - 1) * (number_of_local_matrix_rows)]);
        if (std::abs(local_output_x_vector_part_[i] - x_old[i + (world.rank() - 1) * (number_of_local_matrix_rows)]) >
            max_delta_x_i) {
          max_delta_x_i =
              std::abs(local_output_x_vector_part_[i] - x_old[i + (world.rank() - 1) * (number_of_local_matrix_rows)]);
        }
      }
      i++;
    }
    boost::mpi::gatherv(world, local_output_x_vector_part_.data(), (int)(local_output_x_vector_part_.size()),
                        output_x_vector_.data(), sendcounts, displacements, 0);
    boost::mpi::broadcast(world, output_x_vector_.data(), output_x_vector_.size(), 0);
    num_of_iterations++;
  } while (num_of_iterations < Nmax && max_delta_x_i > epsilon);
  return true;
}

bool deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<std::vector<double>*>(taskData->outputs[0])[0] = output_x_vector_;
  }
  return true;
}
