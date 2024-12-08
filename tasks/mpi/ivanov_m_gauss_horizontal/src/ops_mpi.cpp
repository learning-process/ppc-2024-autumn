// Copyright 2024 Ivanov Mike
#include "mpi/ivanov_m_gauss_horizontal/include/ops_mpi.hpp"

bool ivanov_m_gauss_horizontal_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  auto* input_matrix = reinterpret_cast<double*>(taskData->inputs[0]);
  int size_of_matrix = taskData->inputs_count[0];

  number_of_equations = *reinterpret_cast<int*>(taskData->inputs[1]);
  extended_matrix = std::vector<double>(size_of_matrix);

  for (int i = 0; i < size_of_matrix; i++) {
    extended_matrix[i] = input_matrix[i];
  }

  res = std::vector<double>(number_of_equations);
  return determinant(extended_matrix, number_of_equations) >= DELTA;
}

bool ivanov_m_gauss_horizontal_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  auto* input_matrix = reinterpret_cast<double*>(taskData->inputs[0]);
  int size_of_matrix = taskData->inputs_count[0];
  auto input_number_of_equations = *reinterpret_cast<int*>(taskData->inputs[1]);

  if (input_matrix == nullptr) return false;

  if (size_of_matrix != (input_number_of_equations * (input_number_of_equations + 1))) return false;

  return true;
}

bool ivanov_m_gauss_horizontal_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  int main_row;
  for (int active_row = 0; active_row < number_of_equations; active_row++) {
    main_row = find_max_row(extended_matrix, active_row, active_row, number_of_equations, number_of_equations + 1);

    // check when main row is an active row
    if (main_row != active_row) {
      swap_rows(extended_matrix, active_row, main_row, number_of_equations + 1);
    }

    // forward gauss method
    for (int active_column = number_of_equations; active_column >= active_row; active_column--) {
      extended_matrix[get_linear_index(active_row, active_column, number_of_equations + 1)] /=
          extended_matrix[get_linear_index(active_row, active_row, number_of_equations + 1)];
    }

    for (int active_row_calc = active_row + 1; active_row_calc < number_of_equations; active_row_calc++) {
      for (int active_column_calc = number_of_equations; active_column_calc >= active_row; active_column_calc--) {
        extended_matrix[get_linear_index(active_row_calc, active_column_calc, number_of_equations + 1)] -=
            extended_matrix[get_linear_index(active_row, active_column_calc, number_of_equations + 1)] *
            extended_matrix[get_linear_index(active_row_calc, active_row, number_of_equations + 1)];
      }
    }
  }

  // back gauss method
  for (int active_row = number_of_equations - 1; active_row >= 0; active_row--) {
    double tmp_res = 0;
    for (int active_column = number_of_equations - 1; active_column > active_row; active_column--) {
      tmp_res +=
          extended_matrix[get_linear_index(active_row, active_column, number_of_equations + 1)] * res[active_column];
    }
    res[active_row] =
        extended_matrix[get_linear_index(active_row, number_of_equations, number_of_equations + 1)] - tmp_res;
  }

  return true;
}

bool ivanov_m_gauss_horizontal_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  auto* out = reinterpret_cast<double*>(taskData->outputs[0]);

  for (int i = 0; i < number_of_equations; i++) {
    out[i] = res[i];
  }
  return true;
}

bool ivanov_m_gauss_horizontal_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* input_matrix = reinterpret_cast<double*>(taskData->inputs[0]);
    int size_of_matrix = taskData->inputs_count[0];

    number_of_equations = *reinterpret_cast<int*>(taskData->inputs[1]);
    extended_matrix = std::vector<double>(size_of_matrix);

    for (int i = 0; i < size_of_matrix; i++) {
      extended_matrix[i] = input_matrix[i];
    }

    res = std::vector<double>(number_of_equations);
    return determinant(extended_matrix, number_of_equations) >= DELTA;
  }
  return true;
}

bool ivanov_m_gauss_horizontal_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* input_matrix = reinterpret_cast<double*>(taskData->inputs[0]);
    int size_of_matrix = taskData->inputs_count[0];
    auto input_number_of_equations = *reinterpret_cast<int*>(taskData->inputs[1]);

    if (input_matrix == nullptr) return false;

    if (size_of_matrix != (input_number_of_equations * (input_number_of_equations + 1))) return false;
  }
  return true;
}

bool ivanov_m_gauss_horizontal_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int rank = world.rank();  // rank of this process
  int size = world.size();  // number of processes in this communicator

  boost::mpi::broadcast(world, number_of_equations, 0);

  std::vector<int> sizes(size, 0);   // vector which contains number of equations in each process
  std::vector<int> displs(size, 0);  // vector which specifies the displacement (relative to in_values) from which to
                                     // take the outgoing data to process i

  int local_number_of_equations;                          // number of equations in this process
  int local_number_of_columns = number_of_equations + 1;  // number of columns in this process
  int rest_equations;  // the rest number of equations after devision (number_of_equations / size) --> used in rows
                       // distribution

  int main_row_index;                                        // index of main row (or equation)
  std::vector<double> main_row(local_number_of_columns, 0);  // main row (or equation) on this iteration

  std::vector<double> local_matrix;  // container for "line" of each process

  for (int active_row = 0; active_row < number_of_equations; active_row++) {
    if (rank == 0) {
      // searching of row with max_value
      main_row_index =
          find_max_row(extended_matrix, active_row, active_row, number_of_equations, number_of_equations + 1);

      // check when main row is an active row
      if (main_row_index != active_row) {
        swap_rows(extended_matrix, active_row, main_row_index, local_number_of_columns);
      }

      // used to make main element = 1
      for (int i = number_of_equations; i >= active_row; i--) {
        extended_matrix[get_linear_index(active_row, i, local_number_of_columns)] /=
            extended_matrix[get_linear_index(active_row, active_row, local_number_of_columns)];
      }

      // defining the main row
      for (int i = 0; i < local_number_of_columns; i++) {
        main_row[i] = extended_matrix[get_linear_index(active_row, i, local_number_of_columns)];
      }

      // defining rows distribution by processes
      local_number_of_equations = (number_of_equations - active_row - 1) / size;
      rest_equations = (number_of_equations - active_row - 1) % size;

      // clear outdated data of local_matrix amounts
      sizes.clear();

      // creating a vector which contains number of equations in each process
      sizes.insert(sizes.begin(), rest_equations, (local_number_of_equations + 1) * local_number_of_columns);
      sizes.insert(sizes.begin() + rest_equations, size - rest_equations,
                   local_number_of_equations * local_number_of_columns);

      // creating a vector which contains start index of each process
      displs[0] = (active_row + 1) * (number_of_equations + 1);
      for (int i = 1; i < size; i++) {
        displs[i] = displs[i - 1] + sizes[i - 1];
      }

      // counting the number of columns and rows (or equations) in this process
      local_number_of_equations = sizes[0] / local_number_of_columns;
      for (int i = 1; i < size; i++) {
        world.send(i, 0, sizes[i] / local_number_of_columns);
      }
    } else {
      world.recv(0, 0, local_number_of_equations);
    }

    world.barrier();

    // resize the locla matrix
    local_matrix.resize(local_number_of_equations * local_number_of_columns);

    // sending for each process its "line" (sequence or rows)
    boost::mpi::scatterv(world, extended_matrix.data(), sizes, displs, local_matrix.data(), local_matrix.size(), 0);

    // sending main row for each process
    boost::mpi::broadcast(world, main_row, 0);

    // forward gauss method
    for (int active_row_calc = 0; active_row_calc < local_number_of_equations; active_row_calc++) {
      for (int active_column_calc = number_of_equations; active_column_calc >= active_row; active_column_calc--) {
        local_matrix[get_linear_index(active_row_calc, active_column_calc, local_number_of_columns)] -=
            local_matrix[get_linear_index(active_row_calc, active_row, local_number_of_columns)] *
            main_row[active_column_calc];
      }
    }

    // recieving of "lines" into extended matrix
    boost::mpi::gatherv(world, local_matrix.data(), local_matrix.size(), extended_matrix.data(), sizes, displs, 0);
  }

  if (rank == 0) {
    // back gauss method
    for (int active_row = number_of_equations - 1; active_row >= 0; active_row--) {
      double tmp_res = 0;
      for (int active_column = number_of_equations - 1; active_column > active_row; active_column--) {
        tmp_res +=
            extended_matrix[get_linear_index(active_row, active_column, local_number_of_columns)] * res[active_column];
      }
      res[active_row] =
          extended_matrix[get_linear_index(active_row, number_of_equations, local_number_of_columns)] - tmp_res;
    }
  }

  return true;
}

bool ivanov_m_gauss_horizontal_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* out = reinterpret_cast<double*>(taskData->outputs[0]);

    for (int i = 0; i < number_of_equations; i++) {
      out[i] = res[i];
    }
  }
  return true;
}

// functions

int ivanov_m_gauss_horizontal_mpi::get_linear_index(int row, int col, int number_of_columns) {
  return row * number_of_columns + col;
}

void ivanov_m_gauss_horizontal_mpi::swap_rows(std::vector<double>& matrix, int first_row, int second_row,
                                              int number_of_columns) {
  for (int column = 0; column < number_of_columns; column++) {
    std::swap(matrix[get_linear_index(first_row, column, number_of_columns)],
              matrix[get_linear_index(second_row, column, number_of_columns)]);
  }
}

int ivanov_m_gauss_horizontal_mpi::find_max_row(const std::vector<double>& matrix, int source_row, int source_column,
                                                int number_of_rows, int number_of_columns) {
  int max_row = source_row;
  double max_value = matrix[get_linear_index(source_row, source_column, number_of_rows)];
  for (int active_row = source_row; active_row < number_of_rows; active_row++) {
    if (fabs(matrix[get_linear_index(active_row, source_column, number_of_columns)]) > max_value) {
      max_value = fabs(matrix[get_linear_index(active_row, source_column, number_of_columns)]);
      max_row = active_row;
    }
  }
  return max_row;
}

int ivanov_m_gauss_horizontal_mpi::determinant(const std::vector<double>& matrix, int size) {
  std::vector<double> coef_matrix(size * size);
  int size_of_extended_matrix = size * (size + 1);
  double det = 1;  // determinant

  // create a matrix of coefficients using extended matrix
  int j = 0;
  for (int i = 0; i < size_of_extended_matrix; i++) {
    if ((i + 1) % (size + 1) != 0) {
      coef_matrix[j] = matrix[i];
      j++;
    }
  }

  // create a main element
  int main_row;
  int main_value;
  for (int active_row = 0; active_row < size; active_row++) {
    main_row = find_max_row(coef_matrix, active_row, active_row, size, size);
    main_value = coef_matrix[get_linear_index(main_row, active_row, size)];

    // check when main value = 0 => det = 0
    if (main_value < DELTA) {
      return 0;
    }

    // check when main row is an active row
    if (main_row != active_row) {
      swap_rows(coef_matrix, active_row, main_row, size);
    }

    // gauss method
    for (int active_column = size - 1; active_column >= active_row; active_column--) {
      coef_matrix[get_linear_index(active_row, active_column, size)] /=
          coef_matrix[get_linear_index(active_row, active_row, size)];
    }

    for (int active_row_calc = active_row + 1; active_row_calc < size; active_row_calc++) {
      for (int active_column_calc = size - 1; active_column_calc >= active_row; active_column_calc--) {
        coef_matrix[get_linear_index(active_row_calc, active_column_calc, size)] -=
            coef_matrix[get_linear_index(active_row, active_column_calc, size)] *
            coef_matrix[get_linear_index(active_row_calc, active_row, size)];
      }
    }
  }

  // counting determinant
  for (int i = 0; i < size; i++) {
    det *= coef_matrix[get_linear_index(i, i, size)];
  }
  return det;
}