// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kolokolova_d_max_of_row_matrix/include/ops_seq.hpp"

TEST(kolokolova_d_max_of_row_matrix_seq, test_pipeline_run) {
  int count_rows = 200;
  int size_rows = 900;

  // Создание данных (массив с различными значениями)
  std::vector<int> global_mat;
  for (int i = 0; i < count_rows; ++i) {
    for (int j = 0; j < size_rows; ++j) {
      global_mat.push_back(i + j);  // Используем i + j для создания различных значений
    }
  }

  std::vector<int32_t> seq_max_vec(count_rows, 0);  // Вектор для хранения максимальных значений

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
  taskDataSeq->inputs_count.emplace_back(global_mat.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows));
  taskDataSeq->inputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_max_vec.data()));
  taskDataSeq->outputs_count.emplace_back(seq_max_vec.size());

  // Создание задачи
  auto testTaskSequential = std::make_shared<kolokolova_d_max_of_row_matrix_seq::TestTaskSequential>(taskDataSeq);

  // Создание атрибутов производительности
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;  // Количество запусков

  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;  // Конвертация в секунды
  };

  // Создание и инициализация результатов производительности
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создание анализатора производительности
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  // Печать статистики производительности
  ppc::core::Perf::print_perf_statistic(perfResults);

  // Проверка результатов
  for (size_t i = 0; i < seq_max_vec.size(); i++) {
    EXPECT_EQ(seq_max_vec[i],
              int(size_rows + i - 1));  // Проверка, что максимальное значение в каждой строке соответствует
  }
}

TEST(kolokolova_d_max_of_row_matrix_seq, test_task_run) {
  int count_rows = 3000;
  int size_rows = 6000;

  std::vector<int> global_mat(count_rows * size_rows, 0);
  std::vector<int32_t> seq_max_vec(count_rows, 0);  // Вектор для хранения максимальных значений

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
  taskDataSeq->inputs_count.emplace_back(global_mat.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows));
  taskDataSeq->inputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_max_vec.data()));
  taskDataSeq->outputs_count.emplace_back(seq_max_vec.size());

  // Создание задачи
  auto testTaskSequential = std::make_shared<kolokolova_d_max_of_row_matrix_seq::TestTaskSequential>(taskDataSeq);

  // Создание атрибутов производительности
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;  // Количество запусков
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;  // Конвертация в секунды
  };

  // Создание и инициализация результатов производительности
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создание анализатора производительности
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);  // Запуск задачи

  // Печать статистики производительности
  ppc::core::Perf::print_perf_statistic(perfResults);

  // Проверка результатов
  for (size_t i = 0; i < seq_max_vec.size(); i++) {
    EXPECT_EQ(0, seq_max_vec[i]);  // Проверка, что максимальное значение в каждой строке равно 0
  }
}