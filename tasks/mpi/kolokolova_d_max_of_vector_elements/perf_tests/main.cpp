// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kolokolova_d_max_of_vector_elements/include/ops_mpi.hpp"

//TEST(kolokolova_d_max_of_vector_elements_test, test_pipeline_run) {
//  boost::mpi::communicator world;
//  int num_processes = world.size();               // Количество запущенных процессов
//  int size_rows = num_processes * 2000000;
//  std::vector<int> global_max(num_processes, 0);  // Ожидаемый результат
//  std::vector<int> global_mat(size_rows);     // Матрица
//
//  // Create TaskData
//  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//
//  if (world.rank() == 0) {
//    for (int i = 0; i < size_rows; ++i) {
//      global_mat[i] = 1;
//    }
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
//    taskDataPar->inputs_count.emplace_back(global_mat.size());
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
//    taskDataPar->outputs_count.emplace_back(global_max.size());
//  }
//  
//  auto testMpiTaskParallel = std::make_shared<kolokolova_d_max_of_vector_elements_mpi::TestMPITaskParallel>(taskDataPar, num_processes);
//  ASSERT_EQ(testMpiTaskParallel->validation(), true);
//  testMpiTaskParallel->pre_processing();
//  testMpiTaskParallel->run();
//  testMpiTaskParallel->post_processing();
//
//  // Create Perf attributes
//  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
//  perfAttr->num_running = 10;
//
//  const auto t0 = std::chrono::high_resolution_clock::now();
//  perfAttr->current_timer = [&] {
//    auto current_time_point = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
//    return static_cast<double>(duration) * 1e-9;  // Конвертация в секунды
//  };
//  // Создание и инициализация результатов производительности
//  auto perfResults = std::make_shared<ppc::core::PerfResults>();
//
//  // Создание анализатора производительности
//  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
//  perfAnalyzer->pipeline_run(perfAttr, perfResults);
//
//  // Печать статистики производительности
//  ppc::core::Perf::print_perf_statistic(perfResults);
//
//   if (world.rank() == 0) {
//     std::vector<int> results(num_processes);
//     std::memcpy(results.data(), taskDataPar->outputs.data(), num_processes * sizeof(int));
//
//    // Ожидаемая проверка результатов
//    for (int i = 0; i < num_processes; ++i) {
//      EXPECT_EQ(results[i], 1);
//    }
//   }
//}
//
//TEST(kolokolova_d_max_of_vector_elements_test, test_task_run) {
//  boost::mpi::communicator world;
//
//  int num_processes = world.size();         // Количество запущенных процессов
//  int size_rows = num_processes * 1800000;  // Общее количество элементов
//
//  // Вектор для хранения максимума, ожидаемого для каждого процесса
//  std::vector<int> global_max(num_processes);
//
//  // Вектор для хранения входных данных
//  std::vector<int> global_mat(size_rows);
//
//  // Создание TaskData
//  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//
//  // Заполнение входных данных на процессе 0
//  if (world.rank() == 0) {
//    for (int i = 0; i < size_rows; ++i) {
//      global_mat[i] = i;  // Заполняем последовательными числами
//    }
//
//    // Установка ожидаемых значений максимума
//    for (int i = 0; i < num_processes; ++i) {
//      global_max[i] = int((i + 1) * 1800000 - 1);  // Ожидаем максимума в i-ой группе значений
//    }
//
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
//    taskDataPar->inputs_count.emplace_back(global_mat.size());
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
//    taskDataPar->outputs_count.emplace_back(global_max.size());
//  }
//
//  auto testMpiTaskParallel =
//      std::make_shared<kolokolova_d_max_of_vector_elements_mpi::TestMPITaskParallel>(taskDataPar, num_processes);
//
//  ASSERT_EQ(testMpiTaskParallel->validation(), true);
//
//  testMpiTaskParallel->pre_processing();
//  testMpiTaskParallel->run();
//  testMpiTaskParallel->post_processing();
//
//  // Создание атрибутов производительности
//  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
//  perfAttr->num_running = 10;
//
//  const auto t0 = std::chrono::high_resolution_clock::now();
//
//  perfAttr->current_timer = [&] {
//    auto current_time_point = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
//    return static_cast<double>(duration) * 1e-9;  // Конвертация в секунды
//  };
//
//  // Создание и инициализация результатов производительности
//  auto perfResults = std::make_shared<ppc::core::PerfResults>();
//
//  // Создание анализатора производительности
//  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
//  perfAnalyzer->pipeline_run(perfAttr, perfResults);
//
//  // Печать статистики производительности
//  ppc::core::Perf::print_perf_statistic(perfResults);
//
//  if (world.rank() == 0) {
//    std::vector<int> results(num_processes);
//    std::memcpy(results.data(), taskDataPar->outputs.data(), num_processes * sizeof(int));
//
//    // Ожидаемая проверка результатов
//    for (int i = 0; i < num_processes; ++i) {
//      EXPECT_EQ(results[i], global_max[i]);  // Сравниваем с ожидаемыми значениями
//    }
//  }
//}