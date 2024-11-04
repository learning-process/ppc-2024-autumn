// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/naumov_b_min_colum_matrix/include/ops_mpi.hpp"

TEST(naumov_b_min_colum_matrix_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  const int rows = 40;
  const int cols = 60;
  std::vector<int> global_matrix;
  std::vector<int> global_minima(cols, std::numeric_limits<int>::max());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = naumov_b_min_colum_matrix_mpi::getRandomVector(cols * rows);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_minima.data()));
    taskDataPar->outputs_count.emplace_back(global_minima.size());
  }

  auto minColumnParallel = std::make_shared<naumov_b_min_colum_matrix_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_TRUE(minColumnParallel->validation());
  minColumnParallel->pre_processing();
  minColumnParallel->run();
  minColumnParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(minColumnParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(global_minima.size(), cols);
  }
}

TEST(naumov_b_min_colum_matrix_mpi, test_task_run) {
  boost::mpi::communicator world;
  const int rows = 40;
  const int cols = 60;
  std::vector<int> global_matrix;
  std::vector<int> global_minima(cols, std::numeric_limits<int>::max());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = naumov_b_min_colum_matrix_mpi::getRandomVector(cols * rows);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_minima.data()));
    taskDataPar->outputs_count.emplace_back(global_minima.size());
  }

  auto minColumnParallel = std::make_shared<naumov_b_min_colum_matrix_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_TRUE(minColumnParallel->validation());
  minColumnParallel->pre_processing();
  minColumnParallel->run();
  minColumnParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(minColumnParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    for (int col = 0; col < cols; ++col) {
      ASSERT_NE(global_minima[col], std::numeric_limits<int>::max());
    }
  }
}
