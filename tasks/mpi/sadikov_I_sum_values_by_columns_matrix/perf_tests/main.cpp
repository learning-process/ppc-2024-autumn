#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <iostream>

#include "core/perf/include/perf.hpp"
#include "mpi/sadikov_I_sum_values_by_columns_matrix/include/ops_mpi.h"

TEST(ParallelOperations, mpi_pipline_run) {
  boost::mpi::communicator world;
  const int columns = 3000;
  const int rows = 3000;
  std::vector<int> in(columns * rows, 1);
  std::vector<int> in_index{rows, columns};
  std::vector<int> out_par(columns, 0);
  std::vector<int> answer(columns, columns);
  auto taskData_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData_par = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out_par);
  }
  auto sv_par = std::make_shared<sadikov_I_Sum_values_by_columns_matrix_mpi::MPITaskParallel>(taskData_par);
  ASSERT_EQ(sv_par->validation(), true);
  sv_par->pre_processing();
  sv_par->run();
  sv_par->post_processing();
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sv_par);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(answer, out_par);
  }
}

TEST(ParallelOperations, mpi_task_run) {
  boost::mpi::communicator world;
  const int columns = 3000;
  const int rows = 3000;
  std::vector<int> in(columns * rows, 1);
  std::vector<int> in_index{rows, columns};
  std::vector<int> out_par(columns, 0);
  std::vector<int> answer(columns, columns);
  auto taskData_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData_par = sadikov_I_Sum_values_by_columns_matrix_mpi::CreateTaskData(in, in_index, out_par);
  }
  auto sv_par = std::make_shared<sadikov_I_Sum_values_by_columns_matrix_mpi::MPITaskParallel>(taskData_par);
  ASSERT_EQ(sv_par->validation(), true);
  sv_par->pre_processing();
  sv_par->run();
  sv_par->post_processing();
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sv_par);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(answer, out_par);
  }
}