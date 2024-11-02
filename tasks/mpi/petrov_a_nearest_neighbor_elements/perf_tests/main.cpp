// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/petrov_a_nearest_neighbor_elements/include/ops_mpi.hpp"

TEST(mpi_petrov_a_nearest_neighbor_elements_perf_test, test_pipeline_run1) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);  // Изменен размер для хранения пары значений

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec = {87, 23, 56, 98, 12, 34, 76, 54, 19, 65, 43, 89, 21, 67, 92,
                                         45, 10, 38, 74, 57, 29, 83, 11, 40, 77, 62, 31, 50, 15, 99};
    ;  // Используем случайные данные
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<petrov_a_nearest_neighbor_elements_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Настройка Perf атрибутов
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Инициализация результатов производительности
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Анализ производительности
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  // Проверка результата на корневом процессе
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    // Проверка, что ближайшая пара корректно записана
    int expected_first = global_sum[0];
    int expected_second = global_sum[1];
    ASSERT_EQ(expected_first,77);
    ASSERT_EQ(expected_second,62);
  }
}

TEST(mpi_petrov_a_nearest_neighbor_elements_perf_test, test_task_run1) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);  // Изменен размер для хранения пары значений

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec = {87, 23, 56, 98, 12, 34, 76, 54, 19, 65, 43, 89, 21, 67, 92,
                  45, 10, 38, 74, 57, 29, 83, 11, 40, 77, 62, 31, 50, 15, 99};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<petrov_a_nearest_neighbor_elements_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Настройка Perf атрибутов
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Инициализация результатов производительности
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Анализ производительности
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  // Проверка результата на корневом процессе
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    // Проверка, что ближайшая пара корректно записана
    int expected_first = global_sum[0];
    int expected_second = global_sum[1];
    ASSERT_EQ(expected_first, 77);
    ASSERT_EQ(expected_second, 62);
  }
}
