// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/titov_s_simple_iteration/include/ops_seq.hpp"

TEST(titov_s_simple_iteration_seq, test_pipeline_run) {
    // Создание входных данных
    unsigned int rows = 4;
    unsigned int cols = 5; // Последний столбец — это вектор свободных членов
    std::vector<std::vector<float>> input = {{20.9f, 1.2f, 2.1f, 0.9f, 21.7f},
                                                  {1.2f, 21.2f, 1.5f, 2.5f, 27.46f},
                                                  {2.1f, 1.5f, 19.8f, 1.3f, 28.76f},
                                                  {0.9f, 2.5f, 1.3f, 32.1f, 49.72f}};

  float epsilon = 0.001f;
  std::vector<float> expected_output = {0.8f, 1.0f, 1.2f, 1.4f};

    // Выделение памяти под входные и выходные данные
    std::vector<std::unique_ptr<float[]>> input_ptrs(rows);
    for (unsigned int i = 0; i < rows; ++i) {
        input_ptrs[i] = std::make_unique<float[]>(cols);
        std::copy(input[i].begin(), input[i].end(), input_ptrs[i].get());
    }

    std::unique_ptr<float[]> epsilon_ptr = std::make_unique<float[]>(1);
    epsilon_ptr[0] = epsilon;

    std::vector<float> output(rows, 0.0f);

    // Создание TaskData
    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < rows; ++i) {
        taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_ptrs[i].get()));
    }
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(epsilon_ptr.get()));
    taskDataSeq->inputs_count = {rows, cols};
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    taskDataSeq->outputs_count = {rows};

    // Создание задачи
    auto simpleIteration = std::make_shared<titov_s_simple_iteration_seq::SimpleIterationSequential>(taskDataSeq);


    // Создание Perf атрибутов
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perfAttr->current_timer = [&] {
        auto current_time_point = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
        return static_cast<double>(duration) * 1e-9;
    };

    // Создание и инициализация результатов
    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    // Создание анализатора производительности
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(simpleIteration);
    perfAnalyzer->pipeline_run(perfAttr, perfResults);
    ppc::core::Perf::print_perf_statistic(perfResults);

    // Проверка результатов
    for (unsigned int i = 0; i < rows; ++i) {
        ASSERT_NEAR(output[i], expected_output[i], 0.01f) << "Mismatch at index " << i;
    }
}

TEST(titov_s_simple_iteration_seq, test_task_run) {
    // Создание входных данных
    unsigned int rows = 4;
    unsigned int cols = 5;  // Последний столбец — это вектор свободных членов
    std::vector<std::vector<float>> input = {{20.9f, 1.2f, 2.1f, 0.9f, 21.7f},
                                             {1.2f, 21.2f, 1.5f, 2.5f, 27.46f},
                                             {2.1f, 1.5f, 19.8f, 1.3f, 28.76f},
                                             {0.9f, 2.5f, 1.3f, 32.1f, 49.72f}};

    float epsilon = 0.001f;
    std::vector<float> expected_output = {0.8f, 1.0f, 1.2f, 1.4f};

    // Выделение памяти под входные и выходные данные
    std::vector<std::unique_ptr<float[]>> input_ptrs(rows);
    for (unsigned int i = 0; i < rows; ++i) {
        input_ptrs[i] = std::make_unique<float[]>(cols);
        std::copy(input[i].begin(), input[i].end(), input_ptrs[i].get());
    }

    std::unique_ptr<float[]> epsilon_ptr = std::make_unique<float[]>(1);
    epsilon_ptr[0] = epsilon;

    std::vector<float> output(rows, 0.0f);

    // Создание TaskData
    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < rows; ++i) {
        taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_ptrs[i].get()));
    }
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(epsilon_ptr.get()));
    taskDataSeq->inputs_count = {rows, cols};
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    taskDataSeq->outputs_count = {rows};

    // Создание задачи
    auto simpleIteration = std::make_shared<titov_s_simple_iteration_seq::SimpleIterationSequential>(taskDataSeq);

    // Создание Perf атрибутов
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perfAttr->current_timer = [&] {
      auto current_time_point = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0);
      return static_cast<uint64_t>(duration.count());
    };

    // Выполнение задачи
    ASSERT_TRUE(simpleIteration->validation());
    ASSERT_TRUE(simpleIteration->pre_processing());
    ASSERT_TRUE(simpleIteration->run());
    ASSERT_TRUE(simpleIteration->post_processing());

    // Проверка результата
    for (unsigned int i = 0; i < rows; ++i) {
        EXPECT_NEAR(output[i], expected_output[i], epsilon);
    }
}
