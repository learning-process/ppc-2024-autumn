#include <gtest/gtest.h>

#include <algorithm>  // для std::sort
#include <boost/mpi.hpp>
#include <chrono>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/petrov_o_radix_sort_with_simple_merge/include/ops_mpi.hpp"

using namespace petrov_o_radix_sort_with_simple_merge_mpi;

// Тестирование производительности pipeline_run для параллельной версии
TEST(petrov_o_radix_sort_with_simple_merge_mpi, test_pipeline_run_mpi) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  // Только на процессе с рангом 0 инициализируем данные
  std::vector<int> in;
  std::vector<int> out;
  const size_t array_size = 100000;

  if (world.rank() == 0) {
    std::mt19937 rng(42);  // фиксированный сид для воспроизводимости
    std::uniform_int_distribution<int> dist(-1000000, 1000000);

    // Генерация случайных данных
    in.resize(array_size);
    for (size_t i = 0; i < array_size; ++i) {
      in[i] = dist(rng);
    }

    out.resize(in.size(), 0);
  }

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataMPI->inputs_count.emplace_back(in.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataMPI->outputs_count.emplace_back(out.size());
  } else {
    // На других рангах инициализируем пустые данные
    taskDataMPI->inputs_count.emplace_back(0);
    taskDataMPI->outputs_count.emplace_back(0);
  }

  // Создание задачи
  auto testTaskParallel = std::make_shared<TaskParallel>(taskDataMPI);

  // Создание Perf атрибутов
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Создание и инициализация Perf результатов
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создание Perf анализатора
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  // Проверка результата только на rank=0
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    // Создаём копию исходного массива для сравнения
    std::vector<int> expected = in;
    std::sort(expected.begin(), expected.end());

    // Проверяем, что отсортированный результат совпадает с ожидаемым
    ASSERT_EQ(expected, out) << "Pipeline run MPI: Sorted array does not match the expected result.";
  }

  // Синхронизация всех процессов
  world.barrier();
}

// Тестирование производительности task_run для параллельной версии
TEST(petrov_o_radix_sort_with_simple_merge_mpi, test_task_run_mpi) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> in;
  std::vector<int> out;
  const size_t array_size = 100000;

  if (world.rank() == 0) {
    // Инициализация генератора случайных чисел
    std::mt19937 rng(42);  // фиксированный сид для воспроизводимости
    std::uniform_int_distribution<int> dist(-1000000, 1000000);

    // Генерация случайных данных
    in.resize(array_size);
    for (size_t i = 0; i < array_size; ++i) {
      in[i] = dist(rng);
    }

    // Инициализация выходного массива
    out.resize(in.size(), 0);
  }

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataMPI->inputs_count.emplace_back(in.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataMPI->outputs_count.emplace_back(out.size());
  } else {
    // На других рангах инициализируем пустые данные
    taskDataMPI->inputs_count.emplace_back(0);
    taskDataMPI->outputs_count.emplace_back(0);
  }

  // Создание задачи
  auto testTaskParallel = std::make_shared<TaskParallel>(taskDataMPI);

  // Создание Perf атрибутов
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Создание и инициализация Perf результатов
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создание Perf анализатора
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  // Проверка результата только на rank=0
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    // Создаём копию исходного массива для сравнения
    std::vector<int> expected = in;
    std::sort(expected.begin(), expected.end());

    // Проверяем, что отсортированный результат совпадает с ожидаемым
    ASSERT_EQ(expected, out) << "Task run MPI: Sorted array does not match the expected result.";
  }

  // Синхронизация всех процессов
  world.barrier();
}

// Тестирование производительности pipeline_run для последовательной версии
TEST(petrov_o_radix_sort_with_simple_merge_mpi_seq, test_pipeline_run_seq) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    // Создание данных
    std::vector<int> in;
    std::vector<int> out;
    const size_t array_size = 100000;

    std::mt19937 rng(42);  // фиксированный сид
    std::uniform_int_distribution<int> dist(-1000000, 1000000);

    // Генерация случайных данных
    in.resize(array_size);
    for (size_t i = 0; i < array_size; ++i) {
      in[i] = dist(rng);
    }

    // Инициализация выходного массива
    out.resize(in.size(), 0);

    // Создание TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataSeq->outputs_count.emplace_back(out.size());

    // Создание задачи
    auto testTaskSequential = std::make_shared<TaskSequential>(taskDataSeq);

    // Создание Perf атрибутов
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perfAttr->current_timer = [&] {
      auto current_time_point = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
      return static_cast<double>(duration) * 1e-9;
    };

    // Создание и инициализация Perf результатов
    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    // Создание Perf анализатора
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
    perfAnalyzer->pipeline_run(perfAttr, perfResults);
    ppc::core::Perf::print_perf_statistic(perfResults);

    // Проверка результата
    std::vector<int> expected = in;
    std::sort(expected.begin(), expected.end());
    ASSERT_EQ(expected, out) << "Pipeline run sequential: Sorted array does not match the expected result.";
  }
}

// Тестирование производительности task_run для последовательной версии
TEST(petrov_o_radix_sort_with_simple_merge_mpi_seq, test_task_run_seq) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    // Создание данных
    std::vector<int> in;
    std::vector<int> out;
    const size_t array_size = 100000;

    std::mt19937 rng(42);  // фиксированный сид
    std::uniform_int_distribution<int> dist(-1000000, 1000000);

    // Генерация случайных данных
    in.resize(array_size);
    for (size_t i = 0; i < array_size; ++i) {
      in[i] = dist(rng);
    }

    // Инициализация выходного массива
    out.resize(in.size(), 0);

    // Создание TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataSeq->outputs_count.emplace_back(out.size());

    // Создание задачи
    auto testTaskSequential = std::make_shared<TaskSequential>(taskDataSeq);

    // Создание Perf атрибутов
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perfAttr->current_timer = [&] {
      auto current_time_point = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
      return static_cast<double>(duration) * 1e-9;
    };

    // Создание и инициализация Perf результатов
    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    // Создание Perf анализатора
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
    perfAnalyzer->task_run(perfAttr, perfResults);
    ppc::core::Perf::print_perf_statistic(perfResults);

    // Проверка результата
    std::vector<int> expected = in;
    std::sort(expected.begin(), expected.end());
    ASSERT_EQ(expected, out) << "Task run sequential: Sorted array does not match the expected result.";
  }
}