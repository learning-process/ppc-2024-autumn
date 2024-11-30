// Copyright 2023 Nesterov Alexander
#include "mpi/kalinin_d_matrix_mult_hor_a_vert_b/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> kalinin_d_matrix_mult_hor_a_vert_b::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

bool kalinin_d_matrix_mult_hor_a_vert_b::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  // Считываем указатели на входные данные
  input_A = reinterpret_cast<int*>(taskData->inputs[0]);
  input_B = reinterpret_cast<int*>(taskData->inputs[1]);

  // Читаем размеры матриц
  rows_A = taskData->inputs_count[0];     // Число строк A
  columns_A = taskData->inputs_count[1];  // Число столбцов A
  rows_B = taskData->inputs_count[2];     // Число строк B
  columns_B = taskData->inputs_count[3];  // Число столбцов B

  // Проверяем совместимость матриц
  if (columns_A != rows_B) {
    return false;  // Размеры матриц не совместимы для умножения
  }

  // Инициализируем результат
  C.assign(rows_A * columns_B, 0);

  return true;
}

bool kalinin_d_matrix_mult_hor_a_vert_b::TestMPITaskSequential::validation() {
  internal_order_test();
  
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 &&
         taskData->inputs_count[2] > 0 && taskData->inputs_count[3] > 0 &&
         taskData->inputs_count[1] == taskData->inputs_count[2];
}

bool kalinin_d_matrix_mult_hor_a_vert_b::TestMPITaskSequential::run() {
  internal_order_test();

  // Цикл умножения
  for (int i = 0; i < rows_A; ++i) {
    for (int j = 0; j < columns_B; ++j) {
      for (int k = 0; k < columns_A; ++k) {
        // Обновляем результат
        C[i * columns_B + j] += input_A[i * columns_A + k] * input_B[k * columns_B + j];
      }
    }
  }

  return true;
}

bool kalinin_d_matrix_mult_hor_a_vert_b::TestMPITaskSequential::post_processing() {
  internal_order_test();

  // Размер результирующей матрицы
  size_t total_size = rows_A * columns_B;

  // Копируем результат в выходной буфер
  int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);

  for (size_t i = 0; i < total_size; ++i) {
    output_data[i] = C[i];
  }

  return true;
}

bool kalinin_d_matrix_mult_hor_a_vert_b::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if(world.rank() == 0) {

    // Читаем размеры матриц
    rows_A = taskData->inputs_count[0];     // Число строк A
    columns_A = taskData->inputs_count[1];  // Число столбцов A
    rows_B = taskData->inputs_count[2];     // Число строк B
    columns_B = taskData->inputs_count[3];  // Число столбцов B

    input_A = new int[columns_A * rows_A];
    input_B = new int[columns_B * rows_B];
    auto* tmp_ptr_a = reinterpret_cast<int*>(taskData->inputs[0]);
    auto* tmp_ptr_b = reinterpret_cast<int*>(taskData->inputs[1]);

    // Проверяем совместимость матриц
    for (int i = 0; i < columns_A * rows_A; i++) {
      input_A[i] = tmp_ptr_a[i];
    }

    for (int i = 0; i < columns_B * rows_B; i++) {
      input_B[i] = tmp_ptr_b[i];
    }
    C = nullptr;
  }

  return true;
}

bool kalinin_d_matrix_mult_hor_a_vert_b::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
        return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 &&
         taskData->inputs_count[2] > 0 && taskData->inputs_count[3] > 0 &&
         taskData->inputs_count[1] == taskData->inputs_count[2];
  }
  return true;
}

bool kalinin_d_matrix_mult_hor_a_vert_b::TestMPITaskParallel::run() {
    internal_order_test();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Получаем ранг текущего процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Получаем общее количество процессов

    // Передаем размеры матриц
    int dimensions[4];
    if (rank == 0) {
        dimensions[0] = taskData->inputs_count[1];  // columns_A
        dimensions[1] = taskData->inputs_count[0];  // rows_A
        dimensions[2] = taskData->inputs_count[3];  // columns_B
        dimensions[3] = taskData->inputs_count[2];  // rows_B
    }

    MPI_Bcast(dimensions, 4, MPI_INT, 0, MPI_COMM_WORLD);

    column_A = dimensions[0];
    row_A = dimensions[1];
    column_B = dimensions[2];
    row_B = dimensions[3];

    if (column_A != row_B) {
        if (rank == 0) {
            std::cerr << "Matrix dimensions are incompatible for multiplication: "
                      << "A(" << row_A << "x" << column_A << "), "
                      << "B(" << row_B << "x" << column_B << ")." << std::endl;
        }
        return false;
    }

    if (rank != 0) {
        // Выделение памяти для A и B только для не-нулевых процессов
        input_A = new int[column_A * row_A]();
        input_B = new int[column_B * row_B]();
    }

    // Добавим проверку на нулевые указатели
    if (input_A == nullptr || input_B == nullptr) {
        std::cerr << "Error: input_A or input_B is a null pointer!" << std::endl;
        return false;
    }

    // Рассылаем матрицы A и B всем процессам
    MPI_Bcast(input_A, column_A * row_A, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(input_B, column_B * row_B, MPI_INT, 0, MPI_COMM_WORLD);

    // Синхронизация всех процессов после передачи данных
    MPI_Barrier(MPI_COMM_WORLD); // Ожидаем, пока все процессы получат данные

    // Разбиение матрицы A по строкам между процессами
    int* sendcounts = new int[size];
    int* displs = new int[size];
    std::fill(sendcounts, sendcounts + size, 0);
    std::fill(displs, displs + size, 0);

    int rows_per_proc = row_A / size;
    int extra_rows = row_A % size;
    int offset = 0;

    for (int i = 0; i < size; ++i) {
        sendcounts[i] = (rows_per_proc + (i < extra_rows ? 1 : 0)) * column_A;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    auto* local_A = new int[sendcounts[rank]]();
    MPI_Scatterv(input_A, sendcounts, displs, MPI_INT, local_A, sendcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);

    // Синхронизация после завершения scatter
    MPI_Barrier(MPI_COMM_WORLD); // Ожидаем, пока все процессы завершат разбиение

    // Локальный результат
    int local_rows = sendcounts[rank] / column_A;
    auto* local_res = new int[local_rows * column_B];
    std::fill(local_res, local_res + local_rows * column_B, 0);

    // Умножение
    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < column_B; ++j) {
            for (int k = 0; k < column_A; ++k) {
                local_res[i * column_B + j] += local_A[i * column_A + k] * input_B[k * column_B + j];
            }
        }
    }

    // Синхронизация после завершения умножения
    MPI_Barrier(MPI_COMM_WORLD); // Ожидаем, пока все процессы завершат умножение

    // Сбор результатов
    if (rank == 0) {
        C = new int[row_A * column_B];
    }

    int* recvcounts = new int[size];
    int* recvdispls = new int[size];
    offset = 0;

    for (int i = 0; i < size; ++i) {
        recvcounts[i] = (rows_per_proc + (i < extra_rows ? 1 : 0)) * column_B;
        recvdispls[i] = offset;
        offset += recvcounts[i];
    }

    MPI_Gatherv(local_res, local_rows * column_B, MPI_INT, C, recvcounts, recvdispls, MPI_INT, 0, MPI_COMM_WORLD);

    // Освобождение памяти
    delete[] sendcounts;
    delete[] displs;
    delete[] local_A;
    delete[] local_res;
    delete[] recvcounts;
    delete[] recvdispls;

    return true;
}



bool kalinin_d_matrix_mult_hor_a_vert_b::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    // Размер результирующей матрицы
    size_t total_size = rows_A * columns_B;

    // Копируем результат в выходной буфер
    int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);

    for (size_t i = 0; i < total_size; ++i) {
      output_data[i] = C[i];
    }
  }

  return true;
}

