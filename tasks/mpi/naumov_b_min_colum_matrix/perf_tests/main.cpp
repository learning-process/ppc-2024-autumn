// Copyright 2024 Your Name
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/naumov_b_min_colum_matrix/include/ops_mpi.hpp"

TEST(naumov_b_min_colum_matrix_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vector;
  std::vector<int32_t> global_min(1, INT_MAX);
  int ref = INT_MIN;

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int count_elements = 1000;
    global_vector = naumov_b_min_colum_matrix_mpi::getRandomVector(count_elements);
    int index = rand() % count_elements;
    global_vector[index] = ref;

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(count_elements);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  auto testMpiTaskParallel = std::make_shared<naumov_b_min_colum_matrix_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel->validation());  // Убедитесь, что validation возвращает true
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Создание атрибутов производительности
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Создание и инициализация результатов производительности
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создание анализатора производительности
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(ref, global_min[0]);
  }
}

TEST(naumov_b_min_colum_matrix_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vector;
  std::vector<int32_t> global_min(1, INT_MAX);
  int ref = INT_MIN;

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int count_elements = 1000;
    global_vector = naumov_b_min_colum_matrix_mpi::getRandomVector(count_elements);
    int index = rand() % count_elements;
    global_vector[index] = ref;

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(count_elements);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  auto testMpiTaskParallel = std::make_shared<naumov_b_min_colum_matrix_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel->validation());  // Убедитесь, что validation возвращает true
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Создание атрибутов производительности
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Создание и инициализация результатов производительности
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создание анализатора производительности
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(ref, global_min[0]);
  }
}
