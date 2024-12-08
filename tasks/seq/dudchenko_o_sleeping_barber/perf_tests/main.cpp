#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/dudchenko_o_sleeping_barber/include/ops_seq.hpp"

TEST(sleeping_barber_test, test_pipeline_run) {
  // Параметры задачи
  const int clients = 10;
  const int seats = 3;

  // Создание TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(seats);
  taskDataSeq->outputs_count.emplace_back(1);

  auto testTaskSequential = std::make_shared<dudchenko_o_sleeping_barber_seq::TestSleepingBarber>(taskDataSeq);

  // Валидация
  ASSERT_TRUE(testTaskSequential->validation());

  // Основные этапы задачи
  ASSERT_TRUE(testTaskSequential->pre_processing());
  ASSERT_TRUE(testTaskSequential->run());
  ASSERT_TRUE(testTaskSequential->post_processing());

  // Настройка Perf
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Анализ производительности
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  // Проверка результатов
  ASSERT_TRUE(perfResults->valid());  // Успешная обработка производительности
}

TEST(sleeping_barber_test, test_task_run) {
  // Параметры задачи
  const int clients = 10;
  const int seats = 3;

  // Создание TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(seats);
  taskDataSeq->outputs_count.emplace_back(1);

  auto testTaskSequential = std::make_shared<dudchenko_o_sleeping_barber_seq::TestSleepingBarber>(taskDataSeq);

  // Валидация
  ASSERT_TRUE(testTaskSequential->validation());

  // Основные этапы задачи
  ASSERT_TRUE(testTaskSequential->pre_processing());
  ASSERT_TRUE(testTaskSequential->run());
  ASSERT_TRUE(testTaskSequential->post_processing());

  // Настройка Perf
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Анализ производительности
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);

  // Проверка результатов
  ASSERT_TRUE(perfResults->valid());  // Успешная обработка производительности
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
