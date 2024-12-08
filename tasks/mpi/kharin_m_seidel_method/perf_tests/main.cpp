// gauss_seidel_mpi_perf_test.cpp
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>

#include "core/perf/include/perf.hpp"
#include "mpi/kharin_m_seidel_method/include/ops_mpi.hpp"

namespace mpi = boost::mpi;
using namespace kharin_m_seidel_method;

TEST(GaussSeidel_MPI_PerfTest, test_pipeline_run) {
  mpi::environment env;
  mpi::communicator world;

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

  std::vector<double> xPar(N, 0.0);

  // Создаем TaskData для параллельной версии
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs_count.emplace_back(N * N);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataPar->inputs_count.emplace_back(N);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(xPar.data()));
    taskDataPar->outputs_count.emplace_back(N);
  }

  // Инициализируем параллельную задачу Гаусса-Зейделя
  auto gaussSeidelPar = std::make_shared<GaussSeidelParallel>(taskDataPar);
  ASSERT_TRUE(gaussSeidelPar->validation());
  gaussSeidelPar->pre_processing();
  gaussSeidelPar->run();
  gaussSeidelPar->post_processing();

  // Создаем атрибуты для тестирования производительности
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 5;  // Количество запусков для усреднения
  const mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Создаем и инициализируем результаты производительности
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создаем анализатор производительности
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(gaussSeidelPar);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(GaussSeidel_MPI_PerfTest, test_task_run) {
  mpi::environment env;
  mpi::communicator world;

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

  std::vector<double> xPar(N, 0.0);

  // Создаем TaskData для параллельной версии
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs_count.emplace_back(N * N);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
    taskDataPar->inputs_count.emplace_back(N);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(xPar.data()));
    taskDataPar->outputs_count.emplace_back(N);
  }

  // Инициализируем параллельную задачу Гаусса-Зейделя
  auto gaussSeidelPar = std::make_shared<GaussSeidelParallel>(taskDataPar);
  ASSERT_TRUE(gaussSeidelPar->validation());
  gaussSeidelPar->pre_processing();
  gaussSeidelPar->run();
  gaussSeidelPar->post_processing();

  // Создаем атрибуты для тестирования производительности
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 5;  // Количество запусков для усреднения
  const mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Создаем и инициализируем результаты производительности
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создаем анализатор производительности
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(gaussSeidelPar);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}