#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "mpi/burykin_m_strongin/include/ops_mpi.hpp"

TEST(Burykin_M_Strongin_Perf_MPI, Strongin_Method_Pipeline_Run) {
  // Создаем данные
  double x0 = 0.0;
  double x1 = 10000000.0;
  double epsilon = 0.000000001;
  std::vector<double> out(1, 0.0);

  boost::mpi::communicator world;

  // Создаем TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x0));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x1));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataSeq->outputs_count.emplace_back(out.size());
  }

  // Создаем задачу
  auto fn = [](double x) { return std::sqrt(std::abs(x)) * std::sqrt(std::abs(x)); };
  auto testTask = std::make_shared<burykin_m_strongin::StronginParallel>(taskDataSeq, fn);
  ASSERT_EQ(testTask->validation(), true);
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();

  // Создаем атрибуты производительности
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Создаем и инициализируем результаты производительности
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создаем анализатор производительности
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    // Печатаем статистику производительности
    ppc::core::Perf::print_perf_statistic(perfResults);
    // Проверка результата
    double expected_minimum = 625000;
    EXPECT_NEAR(expected_minimum, out[0], epsilon);
  }
}

TEST(Burykin_M_Strongin_Perf_MPI, Strongin_Method_Task_Run) {
  // Создаем данные
  double x0 = 0.0;
  double x1 = 10000000.0;
  double epsilon = 0.000000001;
  std::vector<double> out(1, 0.0);

  boost::mpi::communicator world;

  // Создаем TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x0));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x1));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataSeq->outputs_count.emplace_back(out.size());
  }

  // Создаем задачу
  auto fn = [](double x) { return std::sqrt(std::abs(x)) * std::sqrt(std::abs(x)); };
  auto testTask = std::make_shared<burykin_m_strongin::StronginParallel>(taskDataSeq, fn);
  ASSERT_EQ(testTask->validation(), true);
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();

  // Создаем атрибуты производительности
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Создаем и инициализируем результаты производительности
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создаем анализатор производительности
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    // Печатаем статистику производительности
    ppc::core::Perf::print_perf_statistic(perfResults);
    // Проверка результата
    double expected_minimum = 625000;
    EXPECT_NEAR(expected_minimum, out[0], epsilon);
  }
}