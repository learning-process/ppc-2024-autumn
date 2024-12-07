#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/timer.hpp>
#include <cmath>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kharin_m_radix_double_sort/include/ops_mpi.hpp"

namespace mpi = boost::mpi;
using namespace kharin_m_radix_double_sort;

// Тест производительности с использованием pipeline_run
TEST(RadixSort_MPI_PerfTest, test_pipeline_run) {
  mpi::environment env;
  mpi::communicator world;

  int N = 10000000;

  std::vector<double> inputData;
  if (world.rank() == 0) {
    inputData.resize(N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1e9, 1e9);

    for (int i = 0; i < N; ++i) {
      inputData[i] = dist(gen);
    }
  }

  std::vector<double> xPar(N, 0.0);

  // Создаем TaskData для параллельной версии
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    taskDataPar->inputs_count.emplace_back(N);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(xPar.data()));
    taskDataPar->outputs_count.emplace_back(N);
  }

  // Инициализируем параллельную задачу
  auto radixSortPar = std::make_shared<RadixSortParallel>(taskDataPar);
  ASSERT_TRUE(radixSortPar->validation());
  radixSortPar->pre_processing();
  radixSortPar->run();
  radixSortPar->post_processing();

  // Создаем атрибуты для тестирования производительности
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 5;  // Количество запусков для усреднения
  const mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Создаем и инициализируем результаты производительности
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создаем анализатор производительности и запускаем pipeline_run
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(radixSortPar);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

// Тест производительности с использованием task_run
TEST(RadixSort_MPI_PerfTest, test_task_run) {
  mpi::environment env;
  mpi::communicator world;

  // Аналогично предыдущему тесту, генерируем большой массив
  int N = 1000000;
  std::vector<double> inputData;
  if (world.rank() == 0) {
    inputData.resize(N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1e9, 1e9);

    for (int i = 0; i < N; ++i) {
      inputData[i] = dist(gen);
    }
  }

  std::vector<double> xPar(N, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
    taskDataPar->inputs_count.emplace_back(N);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(xPar.data()));
    taskDataPar->outputs_count.emplace_back(N);
  }

  auto radixSortPar = std::make_shared<RadixSortParallel>(taskDataPar);
  ASSERT_TRUE(radixSortPar->validation());
  radixSortPar->pre_processing();
  radixSortPar->run();
  radixSortPar->post_processing();

  // Настраиваем параметры производительности
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 5;  // Будем выполнять 5 запусков для усреднения
  const mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(radixSortPar);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}