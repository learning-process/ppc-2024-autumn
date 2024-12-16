// Copyright 2024 Ivanov Mike
#include "seq/ivanov_m_gauss_horizontal/include/ops_seq.hpp"

bool ivanov_m_gauss_horizontal_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  auto* input_matrix = reinterpret_cast<double*>(taskData->inputs[0]);
  int size_of_matrix = taskData->inputs_count[0];

  number_of_equations = *reinterpret_cast<int*>(taskData->inputs[1]);
  if (number_of_equations < 1 || size_of_matrix != (number_of_equations * (number_of_equations + 1))) return false;

  extended_matrix = std::vector<double>(size_of_matrix);

  for (int i = 0; i < size_of_matrix; i++) {
    extended_matrix[i] = input_matrix[i];
  }

  res = std::vector<double>(number_of_equations);
  return determinant(extended_matrix, number_of_equations) >= DELTA;
}

bool ivanov_m_gauss_horizontal_seq::TestTaskSequential::validation() {
  internal_order_test();
  return (static_cast<int>(taskData->inputs.size()) == 2 && static_cast<int>(taskData->inputs_count.size()) == 2 &&
          static_cast<int>(taskData->outputs.size()) == 1 && static_cast<int>(taskData->outputs_count.size()) == 1 &&
          taskData->inputs[0] != nullptr);
}

bool ivanov_m_gauss_horizontal_seq::TestTaskSequential::run() {
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

bool ivanov_m_gauss_horizontal_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  auto* out = reinterpret_cast<double*>(taskData->outputs[0]);

  for (int i = 0; i < number_of_equations; i++) {
    out[i] = res[i];
  }
  return true;
}

// functions

int ivanov_m_gauss_horizontal_seq::get_linear_index(int row, int col, int number_of_columns) {
  return row * number_of_columns + col;
}

void ivanov_m_gauss_horizontal_seq::swap_rows(std::vector<double>& matrix, int first_row, int second_row,
                                              int number_of_columns) {
  for (int column = 0; column < number_of_columns; column++) {
    std::swap(matrix[get_linear_index(first_row, column, number_of_columns)],
              matrix[get_linear_index(second_row, column, number_of_columns)]);
  }
}

int ivanov_m_gauss_horizontal_seq::find_max_row(const std::vector<double>& matrix, int source_row, int source_column,
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

int ivanov_m_gauss_horizontal_seq::determinant(const std::vector<double>& matrix, int size) {
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