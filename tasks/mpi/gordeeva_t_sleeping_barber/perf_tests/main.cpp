#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/gordeeva_t_sleeping_barber/include/ops_mpi.hpp"

// Тест для pipeline_run
TEST(gordeeva_t_sleeping_barber_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  const int max_waiting_chairs = 3;
  int global_res = -1;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(max_waiting_chairs);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&global_res));
  taskDataPar->outputs_count.emplace_back(sizeof(global_res));

  auto testMpiTaskParallel = std::make_shared<gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel>(taskDataPar);

  if (world.size() < 3) {
    ASSERT_EQ(testMpiTaskParallel->validation(), false);
  } else {
    ASSERT_EQ(testMpiTaskParallel->validation(), true);
    ASSERT_TRUE(testMpiTaskParallel->pre_processing());

    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = world.size() - 2;  // Число клиентов
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };

    auto perfResults = std::make_shared<ppc::core::PerfResults>();
    ASSERT_TRUE(testMpiTaskParallel->run());
    ASSERT_TRUE(testMpiTaskParallel->post_processing());

    world.barrier();  // Синхронизация процессов

    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
    perfAnalyzer->pipeline_run(perfAttr, perfResults);

    if (world.rank() == 0) {
      ppc::core::Perf::print_perf_statistic(perfResults);
      ASSERT_EQ(global_res, 0);  // Ожидаемый результат
    }
  }
}

// Тест для task_run с несколькими клиентами
TEST(gordeeva_t_sleeping_barber_mpi, test_task_run_with_multiple_clients) {
  boost::mpi::communicator world;

  const int max_waiting_chairs = 3;  // Количество мест для ожидания клиентов
  int global_res = -1;               // Переменная для хранения результата

  // Подготовка данных задачи
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(max_waiting_chairs);  // Количество мест для ожидания
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&global_res));  // Указатель на результат
  taskDataPar->outputs_count.emplace_back(sizeof(global_res));                  // Размер результата

  // Создание объекта задачи
  auto testMpiTaskParallel = std::make_shared<gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel>(taskDataPar);

  // Проверка валидности задачи, если недостаточно процессов
  if (world.size() < 3) {
    ASSERT_EQ(testMpiTaskParallel->validation(), false);  // Если процессов меньше 3, тест не пройдет
  } else {
    ASSERT_EQ(testMpiTaskParallel->validation(), true);  // Если все в порядке, продолжаем
    ASSERT_TRUE(testMpiTaskParallel->pre_processing());  // Выполнение предварительной обработки

    // Настройка атрибутов производительности для отслеживания
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = world.size() - 2;  // Число клиентов (за вычетом барбера и диспетчера)
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };

    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    // Запуск задачи
    ASSERT_TRUE(testMpiTaskParallel->run());              // Запуск задачи
    ASSERT_TRUE(testMpiTaskParallel->post_processing());  // Обработка после выполнения

    // Синхронизация всех процессов
    world.barrier();

    // Анализ производительности
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
    perfAnalyzer->task_run(perfAttr, perfResults);  // Важный момент: заменен `pipeline_run` на `task_run`

    // Проверка результата выполнения только на главном процессе (rank 0)
    if (world.rank() == 0) {
      // Ожидаем, что результат выполнения задачи будет 0
      ASSERT_EQ(global_res, 0);  // Результат должен быть 0 после выполнения задачи

      // Опционально, вывод статистики производительности
      ppc::core::Perf::print_perf_statistic(perfResults);
    }
  }
}
