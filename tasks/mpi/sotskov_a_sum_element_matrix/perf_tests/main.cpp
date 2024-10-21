// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>
#include <boost/mpi/timer.hpp>
#include <vector>
#include "core/perf/include/perf.hpp"
#include "mpi/sotskov_a_sum_element_matrix/include/ops_mpi.hpp" // Включите нужный заголовочный файл для вашей задачи

TEST(sotskov_a_sum_element_matrix_mpi_perf_test, test_pipeline_run) {
    boost::mpi::communicator world;
    std::vector<std::vector<int>> global_matrix; // Двумерный вектор для хранения матрицы
    std::vector<int32_t> global_sum(1, 0); // Результирующий вектор для глобальной суммы
    int rows = 4; // Пример количества строк
    int cols = 5; // Пример количества столбцов

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
        // Генерация случайной матрицы
        global_matrix = sotskov_a_sum_element_matrix_mpi::getRandomMatrix(rows, cols); // Убедитесь, что эта функция возвращает std::vector<std::vector<int>>

        // Создаем одномерный вектор для передачи через MPI
        std::vector<int> flattened_matrix(rows * cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flattened_matrix[i * cols + j] = global_matrix[i][j];
            }
        }

        // Передаем данные в TaskData
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(flattened_matrix.data()));
        taskDataPar->inputs_count.emplace_back(flattened_matrix.size());
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
        taskDataPar->outputs_count.emplace_back(global_sum.size());
    }

    auto testMpiTaskParallel = std::make_shared<sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel>(taskDataPar, "+");
    ASSERT_EQ(testMpiTaskParallel->validation(), true);
    testMpiTaskParallel->pre_processing();
    testMpiTaskParallel->run();
    testMpiTaskParallel->post_processing();

    // Создаем атрибуты производительности
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };

    // Создаем и инициализируем результаты производительности
    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    // Создаем анализатор производительности
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
    perfAnalyzer->pipeline_run(perfAttr, perfResults);
    if (world.rank() == 0) {
        ppc::core::Perf::print_perf_statistic(perfResults);
        int expected_sum = 0;
        for (const auto& row : global_matrix) {
            expected_sum += std::accumulate(row.begin(), row.end(), 0);
        }
        ASSERT_EQ(expected_sum, global_sum[0]); // Проверка корректности суммы
    }
}

TEST(sotskov_a_sum_element_matrix_mpi_perf_test, test_task_run) {
    boost::mpi::communicator world;
    std::vector<std::vector<int>> global_matrix; // Двумерный вектор для хранения матрицы
    std::vector<int32_t> global_sum(1, 0); // Результирующий вектор для глобальной суммы
    int rows = 4; // Пример количества строк
    int cols = 5; // Пример количества столбцов

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
        // Генерация случайной матрицы
        global_matrix = sotskov_a_sum_element_matrix_mpi::getRandomMatrix(rows, cols);

        // Создаем одномерный вектор для передачи через MPI
        std::vector<int> flattened_matrix(rows * cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flattened_matrix[i * cols + j] = global_matrix[i][j];
            }
        }

        // Передаем данные в TaskData
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(flattened_matrix.data()));
        taskDataPar->inputs_count.emplace_back(flattened_matrix.size());
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
        taskDataPar->outputs_count.emplace_back(global_sum.size());
    }

    auto testMpiTaskParallel = std::make_shared<sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel>(taskDataPar, "+");
    ASSERT_EQ(testMpiTaskParallel->validation(), true);
    testMpiTaskParallel->pre_processing();
    testMpiTaskParallel->run();
    testMpiTaskParallel->post_processing();

    // Создаем атрибуты производительности
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };

    // Создаем и инициализируем результаты производительности
    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    // Создаем анализатор производительности
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
    perfAnalyzer->task_run(perfAttr, perfResults);
    if (world.rank() == 0) {
        ppc::core::Perf::print_perf_statistic(perfResults);
        int expected_sum = 0;
        for (const auto& row : global_matrix) {
            expected_sum += std::accumulate(row.begin(), row.end(), 0);
        }
        ASSERT_EQ(expected_sum, global_sum[0]); // Проверка корректности суммы
    }
}
