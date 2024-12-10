#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/shishkarev_a_gaussian_method_horizontal_strip_pattern/include/ops_mpi.hpp"

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  // Подготовка входных данных
  std::vector<std::vector<double>> input_matrix{{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}};
  std::vector<double> input_vector{1, 0, 1};
  std::vector<double> global_result(3, 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_matrix));
    taskData->inputs_count.emplace_back(sizeof(input_matrix));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_vector));
    taskData->inputs_count.emplace_back(sizeof(input_vector));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskData->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel =
      std::make_shared<shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussianHorizontalParallel>(
          taskData);
  ASSERT_EQ(taskParallel->validation(), true);

  // Настройка производительности
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;

  // Используем std::chrono для измерения времени
  auto start_time = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    return elapsed.count();
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Запуск через pipeline_run
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    // Ожидаемый результат
    std::vector<double> expected_result{1.5, 2, 1.5};
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_NEAR(global_result[0], expected_result[0], 1e-6);
    ASSERT_NEAR(global_result[1], expected_result[1], 1e-6);
    ASSERT_NEAR(global_result[2], expected_result[2], 1e-6);
  }
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi, test_task_run) {
  boost::mpi::communicator world;

  // Подготовка входных данных
  std::vector<std::vector<double>> input_matrix{{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}};
  std::vector<double> input_vector{1, 0, 1};
  std::vector<double> global_result(3, 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_matrix));
    taskData->inputs_count.emplace_back(sizeof(input_matrix));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_vector));
    taskData->inputs_count.emplace_back(sizeof(input_vector));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskData->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel =
      std::make_shared<shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi::MPIGaussianHorizontalParallel>(
          taskData);
  ASSERT_EQ(taskParallel->validation(), true);

  // Настройка производительности
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;

  // Используем std::chrono для измерения времени
  auto start_time = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    return elapsed.count();
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Запуск через task_run
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    // Ожидаемый результат
    std::vector<double> expected_result{1.5, 2, 1.5};
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_NEAR(global_result[0], expected_result[0], 1e-6);
    ASSERT_NEAR(global_result[1], expected_result[1], 1e-6);
    ASSERT_NEAR(global_result[2], expected_result[2], 1e-6);
  }
}
