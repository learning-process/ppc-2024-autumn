#include <gtest/gtest.h>

#include <chrono>
#include <random>

#include "core/perf/include/perf.hpp"
#include "seq/kharin_m_seidel_method/include/ops_seq.hpp"

using namespace kharin_m_seidel_method;

TEST(GaussSeidel_Seq_PerfTest, test_pipeline_run) {
  int N = 1000;  // Размер большой матрицы для теста производительности
  double eps = 1e-6;  // Точность вычислений

  // Генерируем большую диагонально доминантную матрицу
  std::vector<double> A(N * N);
  std::vector<double> b(N);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-10.0, 10.0);

  for (int i = 0; i < N; ++i) {
    double sum = 0.0;
    for (int j = 0; j < N; ++j) {
      if (i != j) {
        A[i * N + j] = dis(gen);
        sum += std::abs(A[i * N + j]);
      }
    }
    // Обеспечиваем диагональную доминантность
    A[i * N + i] = sum + std::abs(dis(gen)) + 1.0;
    b[i] = dis(gen);
  }

  // Выделяем память для вектора решений
  std::vector<double> xSeq(N, 0.0);

  // Создаем TaskData для последовательной версии
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskDataSeq->inputs_count.emplace_back(N * N);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  taskDataSeq->inputs_count.emplace_back(N);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
  taskDataSeq->outputs_count.emplace_back(N);

  // Инициализируем последовательную задачу Гаусса-Зейделя
  auto gaussSeidelSeq = std::make_shared<GaussSeidelSequential>(taskDataSeq);
  ASSERT_TRUE(gaussSeidelSeq->validation());
  gaussSeidelSeq->pre_processing();
  gaussSeidelSeq->run();
  gaussSeidelSeq->post_processing();

  // Создаем атрибуты для тестирования производительности
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 5;  // Количество запусков для усреднения

  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;  // Конвертируем в секунды
  };

  // Создаем и инициализируем результаты производительности
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создаем анализатор производительности
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(gaussSeidelSeq);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  // Вывод статистики производительности
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(GaussSeidel_Seq_PerfTest, test_task_run) {
  int N = 1000;  // Размер большой матрицы для теста производительности
  double eps = 1e-6;  // Точность вычислений

  // Генерируем большую диагонально доминантную матрицу
  std::vector<double> A(N * N);
  std::vector<double> b(N);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-10.0, 10.0);

  for (int i = 0; i < N; ++i) {
    double sum = 0.0;
    for (int j = 0; j < N; ++j) {
      if (i != j) {
        A[i * N + j] = dis(gen);
        sum += std::abs(A[i * N + j]);
      }
    }
    // Обеспечиваем диагональную доминантность
    A[i * N + i] = sum + std::abs(dis(gen)) + 1.0;
    b[i] = dis(gen);
  }

  // Выделяем память для вектора решений
  std::vector<double> xSeq(N, 0.0);

  // Создаем TaskData для последовательной версии
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskDataSeq->inputs_count.emplace_back(N * N);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  taskDataSeq->inputs_count.emplace_back(N);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(xSeq.data()));
  taskDataSeq->outputs_count.emplace_back(N);

  // Инициализируем последовательную задачу Гаусса-Зейделя
  auto gaussSeidelSeq = std::make_shared<GaussSeidelSequential>(taskDataSeq);
  ASSERT_TRUE(gaussSeidelSeq->validation());
  gaussSeidelSeq->pre_processing();
  gaussSeidelSeq->run();
  gaussSeidelSeq->post_processing();

  // Создаем атрибуты для тестирования производительности
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 5;  // Количество запусков для усреднения

  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;  // Конвертируем в секунды
  };

  // Создаем и инициализируем результаты производительности
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создаем анализатор производительности
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(gaussSeidelSeq);
  perfAnalyzer->task_run(perfAttr, perfResults);

  // Вывод статистики производительности
  ppc::core::Perf::print_perf_statistic(perfResults);
}