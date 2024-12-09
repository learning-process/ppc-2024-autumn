#include "mpi/deryabin_m_jacobi_iterative_method/include/ops_mpi.hpp"

#include <thread>

bool deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskSequential::
    pre_processing() {
  internal_order_test();
  input_matrix_ = std::vector<double>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  input_right_vector_ = std::vector<double>(taskData->inputs_count[1]);
  auto* tmp_ptr2 = reinterpret_cast<double*>(taskData->inputs[1]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_matrix_[i] = tmp_ptr[i];
    if (i < taskData->inputs_count[1]) {
      input_right_vector_[i] = tmp_ptr2[i];
    }
  }
  output_x_vector_ = std::vector<double>(input_right_vector_.size());
  return true;
}

bool deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskSequential::
    validation() {
  internal_order_test();
  std::vector<double> matrix_ = std::vector<double>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    matrix_[i] = tmp_ptr[i];
  }
  unsigned short i = 0;
  auto lambda = [&](double first, double second) {
    return (std::abs(first) + std::abs(second));
  };
  while (i != sqrt(matrix_.size())) {
    if (i == 0) {
      if (std::abs(matrix_[0]) <=
          std::accumulate(matrix_.begin() + 1,
                          matrix_.begin() + sqrt(matrix_.size()) - 1, 0,
                          lambda)) {
        return false;
      }
    }
    if (i > 0 && i < sqrt(matrix_.size()) - 1) {
      if (std::abs(matrix_[i * (sqrt(matrix_.size()) + 1)]) <=
          std::accumulate(matrix_.begin() + i * sqrt(matrix_.size()),
                          matrix_.begin() + i * (sqrt(matrix_.size()) + 1) - 1,
                          0, lambda) +
              std::accumulate(
                  matrix_.begin() + i * (sqrt(matrix_.size()) + 1) + 1,
                  matrix_.begin() + (i + 1) * sqrt(matrix_.size()) - 1, 0,
                  lambda)) {
        return false;
      }
    }
    if (i == sqrt(matrix_.size()) - 1) {
      if (std::abs(matrix_[i * (sqrt(matrix_.size()) + 1)]) <=
          std::accumulate(matrix_.begin() + i * sqrt(matrix_.size()),
                          matrix_.end() - 1, 0, lambda)) {
        return false;
      }
    }
    i++;
  }
  return taskData->outputs_count[0] == 1;
}

bool deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskSequential::
    run() {
  internal_order_test();
  unsigned short Nmax = 10000, num_of_iterations = 0;
  double epsilon = pow(10, -6), max_delta_x_i = 0;
  std::vector<double> x_old;
  do {
    x_old = output_x_vector_;
    unsigned short i = 0, j;
    double sum;
    while (i != sqrt(input_matrix_.size())) {
      j = 0;
      sum = 0;
      while (j != sqrt(input_matrix_.size())) {
        if (i != j) {
          sum += input_matrix_[i * sqrt(input_matrix_.size()) + j] * x_old[j];
        }
        j++;
      }
      output_x_vector_[i] =
          (input_right_vector_[i] - sum) *
          (1.0 / input_matrix_[i * (sqrt(input_matrix_.size()) + 1)]);
      if (std::abs(output_x_vector_[i] - x_old[i]) > max_delta_x_i) {
        max_delta_x_i = std::abs(output_x_vector_[i] - x_old[i]);
      }
      i++;
    }
    num_of_iterations++;
  } while (num_of_iterations < Nmax && max_delta_x_i > epsilon);
  return true;
}

bool deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskSequential::
    post_processing() {
  internal_order_test();
  reinterpret_cast<std::vector<double>*>(taskData->outputs[0])[0] =
      output_x_vector_;
  return true;
}

bool deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskParallel::
    pre_processing() {
  internal_order_test();
  return true;
}

bool deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskParallel::
    validation() {
  internal_order_test();
  unsigned short number_of_local_matrix_rows = 0;
  unsigned short ostatochnoe_chislo_strock = 0;
  unsigned short n = 0;
  if (world.rank() == 0) {
    n = (int)(sqrt(taskData->inputs_count[0]));
    number_of_local_matrix_rows = n / world.size();
    ostatochnoe_chislo_strock = n % world.size();
  }
  boost::mpi::broadcast(world, number_of_local_matrix_rows, 0);
  boost::mpi::broadcast(world, n, 0);
  output_x_vector_ = std::vector<double>(n);
  local_input_matrix_part_ = std::vector<double>(number_of_local_matrix_rows * n);
  local_input_right_vector_part_ = std::vector<double>(number_of_local_matrix_rows);
  if (world.rank() == 0) {
      auto* tmp_ptr_matr = reinterpret_cast<double*>(taskData->inputs[0]);
      local_input_matrix_part_ = std::vector<double>((number_of_local_matrix_rows + ostatochnoe_chislo_strock) * n);
      auto* tmp_ptr_vec = reinterpret_cast<double*>(taskData->inputs[1]);
      local_input_right_vector_part_ = std::vector<double>(number_of_local_matrix_rows + ostatochnoe_chislo_strock);
      for (unsigned i = 0; i < (number_of_local_matrix_rows + ostatochnoe_chislo_strock) * n; i++) {
        local_input_matrix_part_[i] = tmp_ptr_matr[n * (n - number_of_local_matrix_rows - ostatochnoe_chislo_strock) + i];
        if (i < number_of_local_matrix_rows + ostatochnoe_chislo_strock) {
          local_input_right_vector_part_[i] = tmp_ptr_vec[n - number_of_local_matrix_rows - ostatochnoe_chislo_strock + i];
        }
      }
      for (int proc = 1; proc < world.size(); proc++) {
        world.send(proc, 0, tmp_ptr_matr + (proc - 1) * number_of_local_matrix_rows * n, number_of_local_matrix_rows * n);
        world.send(proc, 0, tmp_ptr_vec + (proc - 1) * number_of_local_matrix_rows, number_of_local_matrix_rows);
      }
  } else {
    world.recv(0, 0, local_input_matrix_part_.data(),
               number_of_local_matrix_rows * n);
    world.recv(0, 0, local_input_right_vector_part_.data(),
               number_of_local_matrix_rows);
  }
  local_output_x_vector_part_ =
      std::vector<double>(local_input_right_vector_part_.size());
  unsigned short i = 0;
  auto lambda = [&](double first, double second) {
    return (std::abs(first) + std::abs(second));
  };
  while (i != local_input_matrix_part_.size() / n) {
    if (world.rank() == 1 && i == 0) {
      if (std::abs(local_input_matrix_part_[0]) <=
          std::accumulate(local_input_matrix_part_.begin() + 1,
                          local_input_matrix_part_.begin() + n - 1, 0,
                          lambda)) {
        return false;
      }
    }
    if (world.rank() == 0) {
      if (i == number_of_local_matrix_rows + ostatochnoe_chislo_strock - 1) {
        if (std::abs(local_input_matrix_part_[(i + 1) * n - 1]) <=
            std::accumulate(local_input_matrix_part_.begin() + i * n,
                            local_input_matrix_part_.end() - 1, 0, lambda)) {
          return false;
        }
      } else {
        if (std::abs(
                local_input_matrix_part_[(i + 1) * n -
                                         (number_of_local_matrix_rows +
                                          ostatochnoe_chislo_strock - i)]) <=
            std::accumulate(local_input_matrix_part_.begin() + i * n,
                            local_input_matrix_part_.begin() + (i + 1) * n -
                                (number_of_local_matrix_rows +
                                 ostatochnoe_chislo_strock - i) -
                                1,
                            0, lambda) +
                std::accumulate(
                    local_input_matrix_part_.begin() + (i + 1) * n -
                        (number_of_local_matrix_rows +
                         ostatochnoe_chislo_strock - i) +
                        1,
                    local_input_matrix_part_.begin() + (i + 1) * n - 1, 0,
                    lambda)) {
          return false;
        }
      }
    }
    if (world.rank() != 0 && (world.rank() != 1 || i != 0)) {
      if (std::abs(
              local_input_matrix_part_[i * (n + 1) +
                                       (world.rank() - 1) *
                                           (number_of_local_matrix_rows)]) <=
          std::accumulate(
              local_input_matrix_part_.begin() + i * n,
              local_input_matrix_part_.begin() + i * (n + 1) +
                  (world.rank() - 1) * (number_of_local_matrix_rows)-1,
              0, lambda) +
              std::accumulate(
                  local_input_matrix_part_.begin() + i * (n + 1) +
                      (world.rank() - 1) * (number_of_local_matrix_rows) + 1,
                  local_input_matrix_part_.begin() + (i + 1) * n - 1, 0,
                  lambda)) {
        return false;
      }
    }
    i++;
  }
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskParallel::
    run() {
  internal_order_test();
  unsigned short number_of_local_matrix_rows = 0;
  unsigned short ostatochnoe_chislo_strock = 0;
  unsigned short n = 0;
  if (world.rank() == 0) {
    n = (int)(sqrt(taskData->inputs_count[0]));
    number_of_local_matrix_rows = n / world.size();
    ostatochnoe_chislo_strock = n % world.size();
  }
  boost::mpi::broadcast(world, number_of_local_matrix_rows, 0);
  std::vector<int> sendcounts(world.size(), number_of_local_matrix_rows);
  std::vector<int> displacements(world.size());
  boost::mpi::broadcast(world, n, 0);
  if (world.rank() == 0) {
      sendcounts[world.rank()] = number_of_local_matrix_rows + ostatochnoe_chislo_strock;
      displacements[world.rank()] = n - number_of_local_matrix_rows - ostatochnoe_chislo_strock;
  } else {
      displacements[world.rank()] = number_of_local_matrix_rows * (world.rank() - 1) + number_of_local_matrix_rows;
  }
  unsigned short Nmax = 10000, num_of_iterations = 0;
  double epsilon = pow(10, -6), max_delta_x_i = 0;
  std::vector<double> x_old;
  do {
    x_old = output_x_vector_;
    unsigned short i = 0, j;
    double sum;
    while (i != local_output_x_vector_part_.size()) {
      j = 0;
      sum = 0;
      if (world.rank() == 0) {
        while (j != n) {
          if (n - (number_of_local_matrix_rows + ostatochnoe_chislo_strock -
                   i) !=
              j) {
            sum += local_input_matrix_part_[i * n + j] * x_old[j];
          }
          j++;
        }
        local_output_x_vector_part_[i] =
            (local_input_right_vector_part_[i] - sum) *
            (1.0 / local_input_matrix_part_[(i + 1) * n -
                                            (number_of_local_matrix_rows +
                                             ostatochnoe_chislo_strock - i)]);
        if (std::abs(local_output_x_vector_part_[i] -
                     x_old[n - (number_of_local_matrix_rows +
                                ostatochnoe_chislo_strock - i)]) >
            max_delta_x_i) {
          max_delta_x_i = std::abs(local_output_x_vector_part_[i] -
                                   x_old[n - (number_of_local_matrix_rows +
                                              ostatochnoe_chislo_strock - i)]);
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
            (1.0 / local_input_matrix_part_[i * (n + 1) +
                                            (world.rank() - 1) *
                                                (number_of_local_matrix_rows)]);
        if (std::abs(
                local_output_x_vector_part_[i] -
                x_old[i + (world.rank() - 1) * (number_of_local_matrix_rows)]) >
            max_delta_x_i) {
          max_delta_x_i = std::abs(
              local_output_x_vector_part_[i] -
              x_old[i + (world.rank() - 1) * (number_of_local_matrix_rows)]);
        }
      }
      i++;
    }
    boost::mpi::gatherv(world, local_output_x_vector_part_.data(),
                        (int)(local_output_x_vector_part_.size()),
                        output_x_vector_.data(), sendcounts, displacements, 0);
    boost::mpi::broadcast(world, output_x_vector_.data(),
                          output_x_vector_.size(), 0);
    num_of_iterations++;
  } while (num_of_iterations < Nmax && max_delta_x_i > epsilon);
  return true;
}

bool deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskParallel::
    post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<std::vector<double>*>(taskData->outputs[0])[0] =
        output_x_vector_;
  }
  return true;
}
