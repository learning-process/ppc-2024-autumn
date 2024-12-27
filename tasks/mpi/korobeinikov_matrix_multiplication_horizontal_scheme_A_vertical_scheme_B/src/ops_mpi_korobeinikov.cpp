// Copyright 2024 Korobeinikov Arseny
#include "mpi/korobeinikov_matrix_multiplication_horizontal_scheme_A_vertical_scheme_B/include/ops_mpi_korobeinikov.hpp"

#include <algorithm>
#include <functional>

bool korobeinikov_a_test_task_mpi_lab_02::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  A.data.reserve(taskData->inputs_count[0]);
  auto *tmp_ptr_1 = reinterpret_cast<int *>(taskData->inputs[0]);
  std::copy(tmp_ptr_1, tmp_ptr_1 + taskData->inputs_count[0], A.data.begin());
  A.count_rows = (int)*taskData->inputs[1];
  A.count_cols = (int)*taskData->inputs[2];

  B.data.reserve(taskData->inputs_count[3]);
  auto *tmp_ptr_2 = reinterpret_cast<int *>(taskData->inputs[3]);
  std::copy(tmp_ptr_2, tmp_ptr_2 + taskData->inputs_count[3], B.data.begin());
  B.count_rows = (int)*taskData->inputs[4];
  B.count_cols = (int)*taskData->inputs[5];

  res = Matrix(A.count_rows, B.count_cols);
  return true;
}

bool korobeinikov_a_test_task_mpi_lab_02::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs.size() == 6 && taskData->inputs_count.size() == 6 && taskData->outputs.size() == 3 &&
         taskData->outputs_count.size() == 3 && (*taskData->inputs[2] == *taskData->inputs[4]) &&
         ((*taskData->inputs[1]) * (*taskData->inputs[2]) == (int)taskData->inputs_count[0]) &&
         ((*taskData->inputs[4]) * (*taskData->inputs[5]) == (int)taskData->inputs_count[3]);
}

bool korobeinikov_a_test_task_mpi_lab_02::TestMPITaskSequential::run() {
  internal_order_test();

  for (int i = 0; i < A.count_rows; i++) {
    for (int j = 0; j < B.count_cols; j++) {
      res.get_el(i, j) = 0;
      for (int k = 0; k < A.count_cols; k++) {
        res.get_el(i, j) += A.get_el(i, k) * B.get_el(k, j);
      }
    }
  }
  return true;
}

bool korobeinikov_a_test_task_mpi_lab_02::TestMPITaskSequential::post_processing() {
  internal_order_test();

  std::copy(res.data.begin(), res.data.end(), reinterpret_cast<int *>(taskData->outputs[0]));
  *reinterpret_cast<int *>(taskData->outputs[1]) = res.count_rows;
  *reinterpret_cast<int *>(taskData->outputs[2]) = res.count_cols;
  return true;
}

bool korobeinikov_a_test_task_mpi_lab_02::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  return true;
}

bool korobeinikov_a_test_task_mpi_lab_02::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs_count[0] == 0 || taskData->inputs_count[3] == 0) {
      return true;
    }
    return taskData->inputs.size() == 6 && taskData->inputs_count.size() == 6 && taskData->outputs.size() == 3 &&
           taskData->outputs_count.size() == 3 && (*taskData->inputs[2] == *taskData->inputs[4]) &&
           ((*taskData->inputs[1]) * (*taskData->inputs[2]) == (int)taskData->inputs_count[0]) &&
           ((*taskData->inputs[4]) * (*taskData->inputs[5]) == (int)taskData->inputs_count[3]);
  }
  return true;
}

bool korobeinikov_a_test_task_mpi_lab_02::TestMPITaskParallel::run() {
  internal_order_test();
  int num_use_proc;
  int count_rows_on_proc;
  int count_cols_on_proc;
  int A_rows;
  int A_cols;
  int B_rows;
  int B_cols;

  // Getting data by a null process
  if (world.rank() == 0) {
    A.data.reserve(taskData->inputs_count[0]);
    auto *tmp_ptr_1 = reinterpret_cast<int *>(taskData->inputs[0]);
    std::copy(tmp_ptr_1, tmp_ptr_1 + taskData->inputs_count[0], std::back_inserter(A.data));
    A.count_rows = (int)*taskData->inputs[1];
    A.count_cols = (int)*taskData->inputs[2];

    A_rows = A.count_rows;
    A_cols = A.count_cols;

    B.data.reserve(taskData->inputs_count[3]);
    auto *tmp_ptr_2 = reinterpret_cast<int *>(taskData->inputs[3]);
    std::copy(tmp_ptr_2, tmp_ptr_2 + taskData->inputs_count[3], std::back_inserter(B.data));
    B.count_rows = (int)*taskData->inputs[4];
    B.count_cols = (int)*taskData->inputs[5];

    B_rows = B.count_rows;
    B_cols = B.count_cols;

    num_use_proc = std::min(A.count_rows, std::min(world.size(), B.count_cols));
    if (num_use_proc != 0) {
      count_rows_on_proc = A.count_rows / num_use_proc;
      count_cols_on_proc = B.count_cols / num_use_proc;
    } else {
      count_rows_on_proc = 0;
      count_cols_on_proc = 0;
    }

    res = Matrix(A.count_rows, B.count_cols);
  }

  broadcast(world, num_use_proc, 0);
  broadcast(world, count_rows_on_proc, 0);
  broadcast(world, count_cols_on_proc, 0);
  broadcast(world, A_rows, 0);
  broadcast(world, A_cols, 0);
  broadcast(world, B_rows, 0);
  broadcast(world, B_cols, 0);

  if (num_use_proc == 0) {
    return true;
  }
  std::vector<int> displs;
  std::vector<int> scounts;

  // Send A rows by scatterv
  if (world.rank() == 0) {
    for (int i = 0; i < num_use_proc - 1; i++) {
      scounts.push_back(count_rows_on_proc * A_cols);
      displs.push_back(count_rows_on_proc * A_cols * i);
    }
    scounts.push_back(count_rows_on_proc * A_cols + (A_rows % num_use_proc) * A_cols);
    displs.push_back(count_rows_on_proc * A_cols * (num_use_proc - 1));
    for (int i = num_use_proc; i < world.size(); i++) {
      scounts.push_back(0);
      displs.push_back(scounts[num_use_proc - 1] + count_rows_on_proc * A_cols + (A_rows % num_use_proc) * A_cols);
    }
  }

  for (int i = 0; i < num_use_proc - 1; i++) {
    if (world.rank() == i) {
      local_A_rows = std::vector<int>(count_rows_on_proc * A_cols);
    }
  }
  if (world.rank() == num_use_proc - 1) {
    local_A_rows = std::vector<int>(count_rows_on_proc * A_cols + (A_rows % num_use_proc) * A_cols);
  }
  for (int i = num_use_proc; i < world.size(); i++) {
    if (world.rank() == i) {
      local_A_rows = std::vector<int>(1);
    }
  }

  if (world.rank() == 0) {
    scatterv(world, A.data.data(), scounts, displs, local_A_rows.data(), local_A_rows.size(), 0);
  }
  for (int i = 1; i < num_use_proc; i++) {
    if (world.rank() == i) {
      scatterv(world, local_A_rows.data(), local_A_rows.size(), 0);
    }
  }
  for (int i = num_use_proc; i < world.size(); i++) {
    if (world.rank() == i) {
      scatterv(world, local_A_rows.data(), 0, 0);
    }
  }

  //  Send B cols by (send && recv)
  std::vector<int> a_begin_row(num_use_proc);
  std::vector<int> a_end_row(num_use_proc);
  for (int i = 0; i < num_use_proc; i++) {
    a_begin_row[i] = i * count_rows_on_proc;
  }
  for (int i = 0; i < num_use_proc - 1; i++) {
    a_end_row[i] = (i + 1) * count_rows_on_proc - 1;
  }
  a_end_row[num_use_proc - 1] = A_rows - 1;

  std::vector<int> b_begin_col(num_use_proc);
  std::vector<int> b_end_col(num_use_proc);
  for (int i = 0; i < num_use_proc; i++) {
    b_begin_col[i] = i * count_cols_on_proc;
  }
  for (int i = 0; i < num_use_proc - 1; i++) {
    b_end_col[i] = (i + 1) * count_cols_on_proc - 1;
  }
  b_end_col[num_use_proc - 1] = B_cols - 1;

  int delta_for_B = count_cols_on_proc * B_rows;
  std::vector<int> tmp_B_cols;
  if (world.rank() == 0) {
    local_B_cols = std::vector<int>();
    for (int j = 0; j <= b_end_col[0]; j++) {
      for (int i = 0; i < B_rows; i++) {
        local_B_cols.push_back(B.get_el(i, j));
      }
    }

    for (int proc = 1; proc < num_use_proc - 1; proc++) {
      tmp_B_cols.clear();
      for (int j = b_begin_col[proc]; j <= b_end_col[proc]; j++) {
        for (int i = 0; i < B_rows; i++) {
          tmp_B_cols.push_back(B.get_el(i, j));
        }
      }
      world.send(proc, 0, tmp_B_cols.data(), delta_for_B);
    }
    if (num_use_proc != 1) {
      int proc = num_use_proc - 1;
      tmp_B_cols.clear();
      for (int j = b_begin_col[proc]; j <= b_end_col[proc]; j++) {
        for (int i = 0; i < B_rows; i++) {
          tmp_B_cols.push_back(B.get_el(i, j));
        }
      }
      world.send(proc, 0, tmp_B_cols.data(), delta_for_B + (B_cols % num_use_proc) * B_rows);
    }
  } else {
    if (world.rank() == num_use_proc - 1 && num_use_proc != 0) {
      local_B_cols = std::vector<int>(delta_for_B + (B_cols % num_use_proc) * B_rows);
      world.recv(0, 0, local_B_cols.data(), delta_for_B + (B_cols % num_use_proc) * B_rows);
    } else {
      if (world.rank() < num_use_proc) {
        local_B_cols = std::vector<int>(delta_for_B);
        world.recv(0, 0, local_B_cols.data(), delta_for_B);
      }
    }
  }

  // Calculate res matrix
  int size_local_B_cols_for_last = delta_for_B + (B_cols % num_use_proc) * B_rows;
  int size_local_B_cols = delta_for_B;
  tmp_B_cols.clear();
  tmp_B_cols = std::vector<int>(size_local_B_cols_for_last);
  for (int proc1 = 0; proc1 < num_use_proc; proc1++) {
    if (world.rank() == proc1) {
      if (world.rank() != num_use_proc - 1) {
        for (int proc2 = 0; proc2 < num_use_proc; proc2++) {
          if (world.rank() != proc2) {
            world.send(proc2, 0, local_B_cols.data(), size_local_B_cols);
          }
        }
      } else {
        for (int proc2 = 0; proc2 < num_use_proc; proc2++) {
          if (world.rank() != proc2) {
            world.send(proc2, 0, local_B_cols.data(), size_local_B_cols_for_last);
          }
        }
      }

      for (int i = a_begin_row[proc1]; i <= a_end_row[proc1]; i++) {
        for (int j = b_begin_col[proc1]; j <= b_end_col[proc1]; j++) {
          int tmp_res_el = 0;
          for (int n = 0; n < A_cols; n++) {
            tmp_res_el += local_A_rows[(i - a_begin_row[proc1]) * A_cols + n] *
                          local_B_cols[(j - b_begin_col[proc1]) * B_rows + n];
          }
          if (proc1 == 0) {
            res.data[i * B_cols + j] = tmp_res_el;
          } else {
            world.send(0, 0, tmp_res_el);
          }
        }
      }
    } else {
      if (world.rank() < num_use_proc) {
        if (proc1 != num_use_proc - 1) {
          world.recv(proc1, 0, tmp_B_cols.data(), size_local_B_cols);
        } else {
          world.recv(proc1, 0, tmp_B_cols.data(), size_local_B_cols_for_last);
        }

        for (int i = a_begin_row[world.rank()]; i <= a_end_row[world.rank()]; i++) {
          for (int j = b_begin_col[proc1]; j <= b_end_col[proc1]; j++) {
            int tmp_res_el = 0;
            for (int n = 0; n < A_cols; n++) {
              tmp_res_el += local_A_rows[(i - a_begin_row[world.rank()]) * A_cols + n] *
                            tmp_B_cols[(j - b_begin_col[proc1]) * B_rows + n];
            }
            if (world.rank() == 0) {
              res.data[i * B_cols + j] = tmp_res_el;
            } else {
              world.send(0, 0, tmp_res_el);
            }
          }
        }
      }
    }
    if (world.rank() == 0) {
      for (int i = a_end_row[0] + 1; i < A_rows; i++) {
        for (int j = b_begin_col[proc1]; j <= b_end_col[proc1]; j++) {
          int tmp_res_el = 0;
          int proc_sender = i / count_rows_on_proc < num_use_proc ? i / count_rows_on_proc : num_use_proc - 1;
          world.recv(proc_sender, 0, tmp_res_el);
          res.data[i * B_cols + j] = tmp_res_el;
        }
      }
    }
  }

  return true;
}

bool korobeinikov_a_test_task_mpi_lab_02::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    std::copy(res.data.begin(), res.data.end(), reinterpret_cast<int *>(taskData->outputs[0]));
    *reinterpret_cast<int *>(taskData->outputs[1]) = res.count_rows;
    *reinterpret_cast<int *>(taskData->outputs[2]) = res.count_cols;
    return true;
  }
  return true;
}

// mpiexec -n 4 mpi_func_tests