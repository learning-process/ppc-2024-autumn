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
  auto dimension = 0;
  unsigned short block_dimension = 0;
  unsigned short block_rows_columns = 0;
  if (world.rank() == 0) {
    dimension = (unsigned short)sqrt(input_matrix_A.size());
    //output_matrix_C = std::vector<double>(dimension * dimension);
    block_rows_columns = (unsigned short)sqrt(world.size());
    block_dimension = dimension / block_rows_columns;
    if (world.size() == 1 || world.size() != pow(block_rows_columns, 2) || dimension % block_rows_columns != 0) {
      //while (i != dimension) {
        //j = 0;
        //while (j != dimension) {
          //k = 0;
          //while (k != dimension) {
            //output_matrix_C[i * dimension + j] += input_matrix_A[i * dimension + k] * input_matrix_B[k * dimension + j];
            //k++;
          //}
          //j++;
        //}
        //i++;
      //}
      return true;
    }
  }
  boost::mpi::broadcast(world, dimension, 0);
  boost::mpi::broadcast(world, block_rows_columns, 0);
  boost::mpi::broadcast(world, block_dimension, 0);
  output_matrix_C = std::vector<double>(dimension * dimension);
  local_input_matrix_A = std::vector<double>(block_dimension * block_dimension);
  local_input_matrix_B = std::vector<double>(block_dimension * block_dimension);
  local_output_matrix_C = std::vector<double>(block_dimension * block_dimension);
  if (world.rank() == 0) {
    k = 0;
    while (k != block_dimension) {
      std::copy(input_matrix_A.data() + k * dimension, input_matrix_A.data() + k * dimension + block_dimension,
                local_input_matrix_A.begin() + k * block_dimension);
      std::copy(input_matrix_B.data() + k * dimension, input_matrix_B.data() + k * dimension + block_dimension,
                local_input_matrix_B.begin() + k * block_dimension);
      k++;
    }
    while (i != block_rows_columns) {
      j = 0;
      while (j != block_rows_columns) {
        if (i != 0 && j != 0) {
          k = 0;
          while (k != block_dimension) {
            if (i == 0) {
              //world.send(i * block_rows_columns + j, k,
                         //input_matrix_A.data() + (i * block_dimension + k) * dimension + j * block_dimension,
                         //block_dimension);
            } else {
              if (i * block_rows_columns + j - i < i * block_rows_columns) {
                //world.send(i * block_rows_columns + j + block_rows_columns - i, k,
                           //input_matrix_A.data() + (i * block_dimension + k) * dimension + j * block_dimension,
                           //block_dimension);
              } else {
                //world.send(i * block_rows_columns + j - i, k,
                           //input_matrix_A.data() + (i * block_dimension + k) * dimension + j * block_dimension,
                           //block_dimension);
              }
            }
            if (j == 0) {
              //world.send(i * block_rows_columns + j, k,
                         //input_matrix_B.data() + (i * block_dimension + k) * dimension + j * block_dimension,
                         //block_dimension);
            } else {
              if ((i - j) * block_rows_columns + j < 0) {
                //world.send((i + block_rows_columns - j) * block_rows_columns + j, k,
                           //input_matrix_A.data() + (i * block_dimension + k) * dimension + j * block_dimension,
                           //block_dimension);
              } else {
                //world.send((i - j) * block_rows_columns + j, k,
                           //input_matrix_A.data() + (i * block_dimension + k) * dimension + j * block_dimension,
                           //block_dimension);
              }
            }
            k++;
          }
        }
        j++;
      }
      i++;
    }
  } else {
    k = 0;
    while (k != block_dimension) {
      //world.recv(0, k, local_input_matrix_A.data() + k * block_dimension, block_dimension);
      //world.recv(0, k, local_input_matrix_B.data() + k * block_dimension, block_dimension);
      k++;
    }
  }
  i = 0;
  while (i != block_dimension) {
    j = 0;
    while (j != block_dimension) {
      k = 0;
      while (k != block_dimension) {
        local_output_matrix_C[i * block_dimension + j] +=
            local_input_matrix_A[i * block_dimension + k] * local_input_matrix_B[k * block_dimension + j];
        k++;
      }
      j++;
    }
    i++;
  }
  //unsigned short p = 0;
  //while (p != block_rows_columns) {
    //if (world.rank() % block_rows_columns == 0) {
      //world.send(world.rank() + block_rows_columns - 1, 0, local_input_matrix_A.data(), block_dimension);
    //} else {
      //world.send(world.rank() - 1, 0, local_input_matrix_A.data(), block_dimension);
    //}
    //if (world.rank() < block_rows_columns) {
      //world.send(world.rank() + block_rows_columns * (block_rows_columns - 1), 0, local_input_matrix_B.data(),
                 //block_dimension);
    //} else {
      //world.send(world.rank() - block_rows_columns, 0, local_input_matrix_B.data(), block_dimension);
    //}
    //if ((world.rank() + 1) % block_rows_columns == 0) {
      //world.recv(world.rank() - block_rows_columns + 1, 0, local_input_matrix_A.data(), block_dimension);
    //} else {
      //world.recv(world.rank() + 1, 0, local_input_matrix_A.data(), block_dimension);
    //}
    //if (world.rank() >= block_rows_columns * (block_rows_columns - 1)) {
      //world.recv(world.rank() - block_rows_columns * (block_rows_columns - 1), 0, local_input_matrix_B.data(),
                 //block_dimension);
    //} else {
      //world.recv(world.rank() + block_rows_columns, 0, local_input_matrix_B.data(), block_dimension);
    //}
    //i = 0;
    //while (i != block_dimension) {
      //j = 0;
      //while (j != block_dimension) {
        //k = 0;
        //while (k != block_dimension) {
          //local_output_matrix_C[i * block_dimension + j] +=
              //local_input_matrix_A[i * block_dimension + k] * local_input_matrix_B[k * block_dimension + j];
          //k++;
        //}
        //j++;
      //}
      //i++;
    //}
    //p++;
  //}
  //boost::mpi::gather(world, local_output_matrix_C.data(), block_dimension * block_dimension, output_matrix_C, 0);
  return true;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<std::vector<double>*>(taskData->outputs[0])[0] = output_matrix_C;
  }
  return true;
}
