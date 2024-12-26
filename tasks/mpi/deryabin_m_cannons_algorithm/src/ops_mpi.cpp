#include "mpi/deryabin_m_cannons_algorithm/include/ops_mpi.hpp"

#include <thread>

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential::pre_processing() {
  internal_order_test();
  input_matrix_A = std::vector<double>(taskData->inputs_count[0]);
  input_matrix_B = std::vector<double>(taskData->inputs_count[1]);
  auto* tmp_ptr_A = reinterpret_cast<double*>(taskData->inputs[0]);
  auto* tmp_ptr_B = reinterpret_cast<double*>(taskData->inputs[1]);
  std::copy(tmp_ptr_A, tmp_ptr_A + taskData->inputs_count[0], input_matrix_A.begin());
  std::copy(tmp_ptr_B, tmp_ptr_B + taskData->inputs_count[1], input_matrix_B.begin());
  output_matrix_C = std::vector<double>(input_matrix_A.size());
  return true;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == taskData->inputs_count[1] &&
         taskData->inputs_count[1] == pow((unsigned short)sqrt(taskData->inputs_count[0]), 2) &&
         taskData->outputs_count[0] == 1;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential::run() {
  internal_order_test();
  unsigned short i = 0;
  unsigned short j;
  unsigned short count;
  auto dimension = (unsigned short)sqrt(input_matrix_A.size());
  while (i != dimension) {
    j = 0;
    while (j != dimension) {
      count = 0;
      while (count != dimension) {
        output_matrix_C[i * dimension + j] +=
            input_matrix_A[i * dimension + count] * input_matrix_B[count * dimension + j];
        count++;
      }
      j++;
    }
    i++;
  }
  return true;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<std::vector<double>*>(taskData->outputs[0])[0] = output_matrix_C;
  return true;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input_matrix_A = std::vector<double>(taskData->inputs_count[0]);
    input_matrix_B = std::vector<double>(taskData->inputs_count[1]);
    auto* tmp_ptr_A = reinterpret_cast<double*>(taskData->inputs[0]);
    auto* tmp_ptr_B = reinterpret_cast<double*>(taskData->inputs[1]);
    std::copy(tmp_ptr_A, tmp_ptr_A + taskData->inputs_count[0], input_matrix_A.begin());
    std::copy(tmp_ptr_B, tmp_ptr_B + taskData->inputs_count[1], input_matrix_B.begin());
  }
  return true;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] == taskData->inputs_count[1] &&
           taskData->inputs_count[1] == pow((unsigned short)sqrt(taskData->inputs_count[0]), 2) &&
           taskData->outputs_count[0] == 1;
  }
  return true;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::run() {
    internal_order_test();
    unsigned short i = 0;
    unsigned short j;
    unsigned short k;
    int dimension = 0;
    unsigned short block_dimension = 0;
    unsigned short block_rows_columns = 0;
    if (world.rank() == 0) {
      dimension = static_cast<int>(sqrt(input_matrix_A.size()));
      output_matrix_C = std::vector<double>(dimension * dimension, 0.0);
      block_rows_columns = static_cast<unsigned short>(sqrt(world.size()));
      block_dimension = dimension / block_rows_columns;
      if (world.size() != static_cast<int>(pow(block_rows_columns, 2)) || dimension % block_rows_columns != 0) {
        while (i != dimension) {
          j = 0;
          while (j != dimension) {
            k = 0;
            while (k != dimension) {
              output_matrix_C[i * dimension + j] += input_matrix_A[i * dimension + k] * input_matrix_B[k * dimension + j];
              k++;
            }
            j++;
          }
          i++;
        }
        return true;
      }
    }
    boost::mpi::broadcast(world, dimension, 0);
    boost::mpi::broadcast(world, block_rows_columns, 0);
    boost::mpi::broadcast(world, block_dimension, 0);
    int grid_row = world.rank() / block_rows_columns;
    int grid_col = world.rank() % block_rows_columns;
    local_input_matrix_A = std::vector<double>(block_dimension * block_dimension, 0.0);
    local_input_matrix_B = std::vector<double>(block_dimension * block_dimension, 0.0);
    local_output_matrix_C = std::vector<double>(block_dimension * block_dimension, 0.0);
    if (world.rank() == 0) {
        for (unsigned short proc = 0; proc < world.size(); ++proc) {
            int proc_row = proc / block_rows_columns;
            int proc_col = proc % block_rows_columns;
            int A_shift = (proc_row * block_dimension);
            int B_shift = (proc_col * block_dimension);
            std::vector<double> block_A(block_dimension * block_dimension);
            for (unsigned short row = 0; row < block_dimension; ++row) {
                std::copy(
                    input_matrix_A.begin() + (A_shift + row) * dimension + B_shift,
                    input_matrix_A.begin() + (A_shift + row) * dimension + B_shift + block_dimension,
                    block_A.begin() + row * block_dimension
                );
            }
            std::vector<double> block_B(block_dimension * block_dimension);
            for (unsigned short row = 0; row < block_dimension; ++row) {
                std::copy(
                    input_matrix_B.begin() + (A_shift + row) * dimension + B_shift,
                    input_matrix_B.begin() + (A_shift + row) * dimension + B_shift + block_dimension,
                    block_B.begin() + row * block_dimension
                );
            }
            if (proc == 0) {
                local_input_matrix_A = block_A;
                local_input_matrix_B = block_B;
            } else {
                world.send(proc, 0, block_A);
                world.send(proc, 1, block_B);
            }
        }
    } else {
        world.recv(0, 0, local_input_matrix_A);
        world.recv(0, 1, local_input_matrix_B);
    }
    int left = (grid_col == 0) ? (grid_col + block_rows_columns - 1) : (grid_col - 1);
    int right = (grid_col + 1) % block_rows_columns;
    int up = (grid_row == 0) ? (grid_row + block_rows_columns - 1) : (grid_row - 1);
    int down = (grid_row + 1) % block_rows_columns;
    for (int i = 0; i < grid_row; ++i) {
        world.sendrecv_replace(local_input_matrix_A.data(), block_dimension * block_dimension, left, 2, right, 2, world, boost::mpi::status());
    }
    for (int i = 0; i < grid_col; ++i) {
        world.sendrecv_replace(local_input_matrix_B.data(), block_dimension * block_dimension, up, 3, down, 3, world, boost::mpi::status());
    }
    for (int step = 0; step < block_rows_columns; ++step) {
        for (unsigned short row = 0; row < block_dimension; ++row) {
            for (unsigned short col = 0; col < block_dimension; ++col) {
                for (unsigned short k = 0; k < block_dimension; ++k) {
                    local_output_matrix_C[row * block_dimension + col] +=
                        local_input_matrix_A[row * block_dimension + k] * local_input_matrix_B[k * block_dimension + col];
                }
            }
        }
        world.sendrecv_replace(local_input_matrix_A.data(), block_dimension * block_dimension, left, 2, right, 2, world, boost::mpi::status());
        world.sendrecv_replace(local_input_matrix_B.data(), block_dimension * block_dimension, up, 3, down, 3, world, boost::mpi::status());
    }
    boost::mpi::gather(world, local_output_matrix_C.data(), block_dimension * block_dimension, output_matrix_C, 0);
    return true;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<std::vector<double>*>(taskData->outputs[0])[0] = output_matrix_C;
  }
  return true;
}
