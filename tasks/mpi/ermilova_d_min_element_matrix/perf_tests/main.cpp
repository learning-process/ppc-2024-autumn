// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/ermilova_d_min_element_matrix/include/ops_mpi.hpp"

TEST(ermilova_d_min_element_matrix_mpi, test_pipeline_run) {
  std::random_device dev;
  std::mt19937 gen(dev());
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min(1, INT_MAX);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int reference_min = -5000;

  if (world.rank() == 0) {
    const int rows = 5000;
    const int cols = 5000;
    const int upper_border = 1000;
    const int lower_border = -1000;

    global_matrix = ermilova_d_min_element_matrix_mpi::getRandomMatrix(rows, cols, upper_border, lower_border);

    int rnd_rows = gen() % rows;
    int rnd_cols = gen() % cols;
    global_matrix[rnd_rows][rnd_cols] = reference_min;

    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  auto testMpiTaskParallel = std::make_shared<ermilova_d_min_element_matrix_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(reference_min, global_min[0]);
  }
}

TEST(ermilova_d_min_element_matrix_mpi, test_task_run) {
  std::random_device dev;
  std::mt19937 gen(dev());
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min(1, INT_MAX);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int reference_min = -5000;

  if (world.rank() == 0) {
    const int rows = 5000;
    const int cols = 5000;
    const int upper_border = 1000;
    const int lower_border = -1000;

    global_matrix = ermilova_d_min_element_matrix_mpi::getRandomMatrix(rows, cols, upper_border, lower_border);
    int rnd_rows = gen() % rows;
    int rnd_cols = gen() % cols;
    global_matrix[rnd_rows][rnd_cols] = reference_min;

    for (unsigned int i = 0; i < global_matrix.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  auto testMpiTaskParallel = std::make_shared<ermilova_d_min_element_matrix_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(reference_min, global_min[0]);
  }
}
