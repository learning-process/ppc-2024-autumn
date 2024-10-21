// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>
#include <numeric> // Для std::accumulate
#include <memory> // Для std::shared_ptr

#include "mpi/sotskov_a_sum_element_matrix/include/ops_mpi.hpp" // Включите заголовочный файл для вашей реализации

// Тестовый случай для суммирования элементов матрицы с использованием MPI
TEST(sotskov_a_sum_element_matrix_mpi, test_sum_3x3_matrix) {
    boost::mpi::communicator world;
    std::vector<std::vector<int>> global_matrix; // Двумерный вектор для хранения матрицы
    std::vector<int32_t> global_sum(1, 0); // Вектор для хранения глобальной суммы
    int rows = 3;
    int cols = 3;

    // Создаем TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
        // Генерация случайной матрицы
        std::cerr << "Rank 0: Генерация матрицы" << std::endl;
        global_matrix = sotskov_a_sum_element_matrix_mpi::getRandomMatrix(rows, cols); // Проверьте, что возвращает этот метод

        // Создаем одномерный вектор для передачи через MPI
        std::vector<int> flattened_matrix(rows * cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flattened_matrix[i * cols + j] = global_matrix[i][j];
            }
        }

        std::cerr << "Rank 0: Передача данных в TaskData" << std::endl;
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(flattened_matrix.data()));
        taskDataPar->inputs_count.emplace_back(flattened_matrix.size());
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
        taskDataPar->outputs_count.emplace_back(global_sum.size());
    }

    // Синхронизируем все процессы перед запуском задачи
    world.barrier();

    // Запускаем параллельную задачу
    std::cerr << "Rank " << world.rank() << ": Начало выполнения параллельной задачи" << std::endl;
    sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    // Синхронизируем после выполнения задачи
    world.barrier();

    if (world.rank() == 0) {
        // Создаем данные для последовательного выполнения
        std::cerr << "Rank 0: Создание данных для последовательного выполнения" << std::endl;
        std::vector<int32_t> reference_sum(1, 0);

        // Создаем TaskData для последовательного выполнения
        std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
        taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[0].data()));  // Исправлено
        taskDataSeq->inputs_count.emplace_back(rows * cols);
        taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
        taskDataSeq->outputs_count.emplace_back(reference_sum.size());

        // Запуск последовательной задачи
        std::cerr << "Rank 0: Начало выполнения последовательной задачи" << std::endl;
        sotskov_a_sum_element_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
        ASSERT_EQ(testMpiTaskSequential.validation(), true);
        testMpiTaskSequential.pre_processing();
        testMpiTaskSequential.run();
        testMpiTaskSequential.post_processing();

        // Сравниваем результаты
        std::cerr << "Rank 0: Сравнение результатов" << std::endl;
        ASSERT_EQ(reference_sum[0], global_sum[0]); // Проверка на соответствие суммы
    }

    // Синхронизация всех процессов перед завершением
    world.barrier();
}

