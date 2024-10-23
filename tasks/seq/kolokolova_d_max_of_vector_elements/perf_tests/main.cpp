// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kolokolova_d_max_of_vector_elements/include/ops_seq.hpp"

TEST(kolokolova_d_max_of_vector_elements_seq, test_pipeline_run) {
    int count_rows = 100;
    int size_rows = 400;

    // Создание данных (массив, заполненный единицами)
    std::vector<int> global_mat(count_rows * size_rows, 1);
    std::vector<int32_t> seq_max_vec(count_rows, 0); // Вектор для хранения максимальных значений

    // Создание TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(global_mat.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows));
    taskDataSeq->inputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_max_vec.data()));
    taskDataSeq->outputs_count.emplace_back(seq_max_vec.size());

    // Создание задачи
    auto testTaskSequential = std::make_shared<kolokolova_d_max_of_vector_elements_seq::TestTaskSequential>(taskDataSeq);

    // Создание атрибутов производительности
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10; // Количество запусков
    const auto t0 = std::chrono::high_resolution_clock::now();
    perfAttr->current_timer = [&] {
        auto current_time_point = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
        return static_cast<double>(duration) * 1e-9; // Конвертация в секунды
    };

    // Создание и инициализация результатов производительности
    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    // Создание анализатора производительности
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
    perfAnalyzer->pipeline_run(perfAttr, perfResults); // Запуск конвейера

    // Печать статистики производительности
    ppc::core::Perf::print_perf_statistic(perfResults);

    // Проверка результатов
    for (size_t i = 0; i < seq_max_vec.size(); i++) {
        EXPECT_EQ(1, seq_max_vec[i]); // Проверка, что максимальное значение в каждой строке равно 1
    }
}

TEST(kolokolova_d_max_of_vector_elements_seq, test_task_run) {
    int count_rows = 100;
    int size_rows = 400;

    // Создание данных (массив, заполненный единицами)
    std::vector<int> global_mat(count_rows * size_rows, 1);
    std::vector<int32_t> seq_max_vec(count_rows, 0); // Вектор для хранения максимальных значений

    // Создание TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(global_mat.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows));
    taskDataSeq->inputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_max_vec.data()));
    taskDataSeq->outputs_count.emplace_back(seq_max_vec.size());

    // Создание задачи
    auto testTaskSequential = std::make_shared<kolokolova_d_max_of_vector_elements_seq::TestTaskSequential>(taskDataSeq);

    // Создание атрибутов производительности
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10; // Количество запусков
    const auto t0 = std::chrono::high_resolution_clock::now();
    perfAttr->current_timer = [&] {
        auto current_time_point = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
        return static_cast<double>(duration) * 1e-9; // Конвертация в секунды
    };

    // Создание и инициализация результатов производительности
    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    // Создание анализатора производительности
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
    perfAnalyzer->task_run(perfAttr, perfResults); // Запуск задачи

    // Печать статистики производительности
    ppc::core::Perf::print_perf_statistic(perfResults);

    // Проверка результатов
    for (size_t i = 0; i < seq_max_vec.size(); i++) {
        EXPECT_EQ(1, seq_max_vec[i]); // Проверка, что максимальное значение в каждой строке равно 1
    }
}