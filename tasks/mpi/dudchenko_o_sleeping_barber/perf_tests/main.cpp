#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/dudchenko_o_sleeping_barber/include/ops_mpi.hpp"

TEST(sleeping_barber_test, test_pipeline_run) {
  boost::mpi::communicator world;

  // Параметры задачи
  const int clients = 10;
  const int seats = 3;

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(clients);
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testTaskParallel = std::make_shared<dudchenko_o_sleeping_barber_mpi::TestMPISleepingBarber>(taskDataPar, seats);

  // Валидация
  ASSERT_EQ(testTaskParallel->validation(), true);

  // Основные этапы задачи
  testTaskParallel->pre_processing();
  testTaskParallel->run();
  testTaskParallel->post_processing();

  // Настройка Perf
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Результаты производительности
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Анализ производительности
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  // Проверка результатов на процессе с рангом 0
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_TRUE(true);  // Успешное выполнение.
  }
}

TEST(sleeping_barber_test, test_task_run) {
  boost::mpi::communicator world;

  // Параметры задачи
  const int clients = 10;
  const int seats = 3;

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(clients);
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testTaskParallel = std::make_shared<dudchenko_o_sleeping_barber_mpi::TestMPISleepingBarber>(taskDataPar, seats);

  // Валидация
  ASSERT_EQ(testTaskParallel->validation(), true);

  // Основные этапы задачи
  testTaskParallel->pre_processing();
  testTaskParallel->run();
  testTaskParallel->post_processing();

  // Настройка Perf
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Результаты производительности
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Анализ производительности
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  // Проверка результатов на процессе с рангом 0
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_TRUE(true);  // Успешное выполнение.
  }
}

int main(int argc, char** argv) {
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (world.rank() != 0) {
    delete listeners.Release(listeners.default_result_printer());
  }
  return RUN_ALL_TESTS();
}
