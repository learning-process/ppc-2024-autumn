// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/yasakova_t_min_of_vector_elements/include/ops_mpi_yasakova.hpp"

TEST(yasakova_t_min_of_vector_elements_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_minimum(1, INT_MAX);
  int ref = INT_MIN;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::random_device dev;
    std::mt19937 gen(dev());

    int count_rows = 4000;
    int count_columns = 4000;
    int gen_minimum = -500;
    int gen_maximum = 500;

    global_matrix = yasakova_t_min_of_vector_elements_mpi::RandomMatrix(count_rows, count_columns, gen_minimum, gen_maximum);
    int index = gen() % (count_rows * count_columns);
    global_matrix[index / count_columns][index / count_rows] = ref;

    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_columns);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_minimum.data()));
    taskDataPar->outputs_count.emplace_back(global_minimum.size());
  }

  auto testMpiTaskParallel = std::make_shared<yasakova_t_min_of_vector_elements_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(ref, global_minimum[0]);
  }
}

TEST(yasakova_t_min_of_vector_elements_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_sum(1, INT_MAX);
  int ref = INT_MIN;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::random_device dev;
    std::mt19937 gen(dev());

    int count_rows = 4000;
    int count_columns = 4000;
    int gen_minimum = -500;
    int gen_maximum = 500;

    global_matrix = yasakova_t_min_of_vector_elements_mpi::RandomMatrix(count_rows, count_columns, gen_minimum, gen_maximum);
    int index = gen() % (count_rows * count_columns);
    global_matrix[index / count_columns][index / count_rows] = ref;

    for (unsigned int i = 0; i < global_matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_columns);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  auto testMpiTaskParallel = std::make_shared<yasakova_t_min_of_vector_elements_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(ref, global_sum[0]);
  }
}