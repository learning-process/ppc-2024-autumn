// Copyright 2023 Nesterov Alexander
#include "mpi/sotskov_a_sum_element_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<std::vector<int>> sotskov_a_sum_element_matrix_mpi::getRandomMatrix(int rows, int cols) {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = gen() % 100; // Генерация случайных чисел от 0 до 99
        }
    }
    
    return matrix;
}

// Реализация методов класса TestMPITaskSequential
bool sotskov_a_sum_element_matrix_mpi::TestMPITaskSequential::pre_processing() {
    internal_order_test();
    
    // Инициализация матрицы
    int rows = taskData->inputs_count[0]; // Количество строк
    int cols = taskData->inputs_count[1]; // Количество столбцов
    input_.resize(rows, std::vector<int>(cols));
    
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < static_cast<unsigned>(rows); i++) {
    for (unsigned j = 0; j < static_cast<unsigned>(cols); j++) {
        input_[i][j] = tmp_ptr[i * cols + j];
    }
}


    // Инициализация значения для вывода
    res = 0;
    return true;
}

bool sotskov_a_sum_element_matrix_mpi::TestMPITaskSequential::validation() {
    internal_order_test();
    return taskData->outputs_count[0] == 1; // Проверка количества элементов на вывод
}

bool sotskov_a_sum_element_matrix_mpi::TestMPITaskSequential::run() {
    internal_order_test();

    // Суммирование элементов матрицы
    for (const auto& row : input_) {
        res += std::accumulate(row.begin(), row.end(), 0);
    }
    return true;
}

bool sotskov_a_sum_element_matrix_mpi::TestMPITaskSequential::post_processing() {
    internal_order_test();
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res; // Сохранение результата
    return true;
}

// Реализация методов класса TestMPITaskParallel
bool sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel::pre_processing() {
    internal_order_test();
    
    int total_rows = taskData->inputs_count[0];
    int total_cols = taskData->inputs_count[1];
    unsigned int delta = total_rows / world.size();
    unsigned int remainder = total_rows % world.size(); // Остаток строк, которые нужно распределить

    if (world.rank() == 0) {
        // Инициализация матрицы на главном процессе
        input_.resize(total_rows, std::vector<int>(total_cols));
        auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
        for (size_t i = 0; i < total_rows; i++) {
    for (size_t j = 0; j < total_cols; j++) {
        input_[i][j] = tmp_ptr[i * total_cols + j];
    }
}


        // Отправка данных другим процессам
        for (size_t proc = 1; proc < static_cast<size_t>(world.size()); proc++) {  // Используйте size_t или unsigned
    size_t rows_to_send = (proc < static_cast<size_t>(remainder)) ? delta + 1 : delta; // Приведение типов
    for (size_t i = 0; i < rows_to_send; i++) {
        world.send(proc, 0, input_[proc * delta + std::min(proc, static_cast<size_t>(remainder)) + i].data(), total_cols);
    }
}

    }

    // Получение локальных данных
    local_input_.resize(delta + (static_cast<size_t>(world.rank()) < remainder ? 1 : 0), std::vector<int>(total_cols));

    
    if (world.rank() == 0) {
    local_input_ = std::vector<std::vector<int>>(input_.begin(), input_.begin() + delta + (world.rank() < static_cast<int>(remainder) ? 1 : 0));
} else {
    for (unsigned i = 0; i < local_input_.size(); i++) {
        world.recv(0, 0, local_input_[i].data(), total_cols);
    }
}


    // Инициализация значения для вывода
    res = 0;
    return true;
}

bool sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel::validation() {
    internal_order_test();
    if (world.rank() == 0) {
        return taskData->outputs_count[0] == 1; // Проверка количества элементов на вывод
    }
    return true;
}

bool sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel::run() {
    internal_order_test();

    int local_res = 0;

    // Суммирование локальных элементов матрицы
    for (const auto& row : local_input_) {
        local_res += std::accumulate(row.begin(), row.end(), 0);
    }

    // Сбор глобальной суммы
    reduce(world, local_res, res, std::plus<int>(), 0);
    return true;
}

bool sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel::post_processing() {
    internal_order_test();
    if (world.rank() == 0) {
        reinterpret_cast<int*>(taskData->outputs[0])[0] = res; // Сохранение результата
    }
    return true;
}
