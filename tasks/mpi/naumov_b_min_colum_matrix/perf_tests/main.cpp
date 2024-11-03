// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <memory>
#include <vector>

#include "mpi/naumov_b_min_colum_matrix/include/ops_mpi.hpp"

using namespace naumov_b_min_colum_matrix_mpi;

// Тест для проверки производительности
// TEST(mpi_naumov_b_min_colum_matrix_perf_test, test_pipeline_run) {
//   boost::mpi::communicator world;
//   std::vector<int> global_vec;
//   std::vector<int32_t> global_sum(1, 0);

//   // Создаем TaskData
//   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//   int count_size_vector;

//   if (world.rank() == 0) {
//     count_size_vector = 120;                              // Размер вектора
//     global_vec = std::vector<int>(count_size_vector, 1);  // Заполняем вектор единицами
//     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//     taskDataPar->inputs_count.emplace_back(global_vec.size());
//     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
//     taskDataPar->outputs_count.emplace_back(global_sum.size());
//   }

//   auto testMpiTaskParallel = std::make_shared<TestMPITaskParallel>(taskDataPar);
//   ASSERT_EQ(testMpiTaskParallel->validation(), true);
//   testMpiTaskParallel->pre_processing();
//   testMpiTaskParallel->run();
//   testMpiTaskParallel->post_processing();

//   // Создаем атрибуты производительности
//   auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
//   perfAttr->num_running = 10;  // Количество запусков для анализа производительности
//   const boost::mpi::timer current_timer;
//   perfAttr->current_timer = [&] { return current_timer.elapsed(); };

//   // Создаем и инициализируем результаты производительности
//   auto perfResults = std::make_shared<ppc::core::PerfResults>();

//   // Создаем анализатор производительности
//   auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
//   perfAnalyzer->pipeline_run(perfAttr, perfResults);

//   if (world.rank() == 0) {
//     ppc::core::Perf::print_perf_statistic(perfResults);
//     ASSERT_EQ(count_size_vector, global_sum[0]);  // Проверка результата
//   }
// }

// TEST(mpi_naumov_b_min_colum_matrix_perf_test, test_task_run) {
//   boost::mpi::communicator world;
//   std::vector<int> global_vec;
//   std::vector<int32_t> global_sum(1, 0);

//   // Создаем TaskData
//   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//   int count_size_vector;

//   if (world.rank() == 0) {
//     count_size_vector = 120;                              // Размер вектора
//     global_vec = std::vector<int>(count_size_vector, 1);  // Заполняем вектор единицами
//     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
//     taskDataPar->inputs_count.emplace_back(global_vec.size());
//     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
//     taskDataPar->outputs_count.emplace_back(global_sum.size());
//   }

//   auto testMpiTaskParallel = std::make_shared<TestMPITaskParallel>(taskDataPar);
//   ASSERT_EQ(testMpiTaskParallel->validation(), true);
//   testMpiTaskParallel->pre_processing();
//   testMpiTaskParallel->run();
//   testMpiTaskParallel->post_processing();

//   // Создаем атрибуты производительности
//   auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
//   perfAttr->num_running = 10;  // Количество запусков для анализа производительности
//   const boost::mpi::timer current_timer;
//   perfAttr->current_timer = [&] { return current_timer.elapsed(); };

//   // Создаем и инициализируем результаты производительности
//   auto perfResults = std::make_shared<ppc::core::PerfResults>();

//   // Создаем анализатор производительности
//   auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
//   perfAnalyzer->task_run(perfAttr, perfResults);

//   if (world.rank() == 0) {
//     ppc::core::Perf::print_perf_statistic(perfResults);
//     ASSERT_EQ(count_size_vector, global_sum[0]);  // Проверка результата
//   }
// }
