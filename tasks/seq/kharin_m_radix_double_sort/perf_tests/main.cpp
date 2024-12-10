#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kharin_m_radix_double_sort/include/ops_seq.hpp"

using kharin_m_radix_double_sort::RadixSortSequential;

TEST(RadixSort_Sequential_PerfTest, test_pipeline_run) {
  // Выберем достаточно большой размер для тестирования производительности
  int N = 10000000;
  std::vector<double> inputData(N);
  std::vector<double> outputData(N, 0.0);

  // Генерируем случайный массив double
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1e9, 1e9);
    for (int i = 0; i < N; ++i) {
      inputData[i] = dist(gen);
    }
  }

  // Создаем TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
  taskDataSeq->inputs_count.emplace_back(N);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputData.data()));
  taskDataSeq->outputs_count.emplace_back(N);

  // Создаем задачу
  auto radixSortSeq = std::make_shared<RadixSortSequential>(taskDataSeq);
  ASSERT_TRUE(radixSortSeq->validation());
  radixSortSeq->pre_processing();
  radixSortSeq->run();
  radixSortSeq->post_processing();

  // Создаем атрибуты для тестирования производительности
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 5;  // Запустим 5 раз для усреднения
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Создаем и инициализируем результаты производительности
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создаем анализатор производительности
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(radixSortSeq);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);

  std::vector<double> refData = inputData;
  std::sort(refData.begin(), refData.end());
  for (int i = 0; i < N; ++i) {
    ASSERT_NEAR(refData[i], outputData[i], 1e-12);
  }
}

TEST(RadixSort_Sequential_PerfTest, test_task_run) {
  // Аналогичный тест, но с task_run
  int N = 1000000;
  std::vector<double> inputData(N);
  std::vector<double> outputData(N, 0.0);

  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1e9, 1e9);
    for (int i = 0; i < N; ++i) {
      inputData[i] = dist(gen);
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
  taskDataSeq->inputs_count.emplace_back(N);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputData.data()));
  taskDataSeq->outputs_count.emplace_back(N);

  auto radixSortSeq = std::make_shared<RadixSortSequential>(taskDataSeq);
  ASSERT_TRUE(radixSortSeq->validation());
  radixSortSeq->pre_processing();
  radixSortSeq->run();
  radixSortSeq->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(radixSortSeq);
  perfAnalyzer->task_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);

  // Проверка корректности результата
  std::vector<double> refData = inputData;
  std::sort(refData.begin(), refData.end());
  for (int i = 0; i < N; ++i) {
    ASSERT_NEAR(refData[i], outputData[i], 1e-12);
  }
}