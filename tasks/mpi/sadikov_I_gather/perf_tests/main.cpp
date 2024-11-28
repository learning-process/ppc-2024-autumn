#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <iostream>

#include "core/perf/include/perf.hpp"
#include "mpi/sadikov_I_gather/include/ops_mpi.h"

TEST(sadikov_I_Sum_values_by_columns_matrix_mpi, mpi_pipline_run) {
  boost::mpi::communicator world;
  const int columns = 4000;
  const int rows = 4000;
  const int root = 0;
  std::vector<int> in;
  std::vector<int> in_index{rows, columns};
  std::vector<int> out_par(columns, 0);
  std::vector<int> answer(columns, columns);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = std::vector<int>(rows * columns, 1);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in_index[0]);
    taskData->inputs_count.emplace_back(in_index[1]);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskData->outputs_count.emplace_back(out_par.size());
  }
  auto sv_par = std::make_shared<sadikov_I_gather_mpi::MPITaskParallel>(taskData);
  sv_par->SetRoot(root);
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
  if (world.rank() == root) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(answer, out_par);
  }
}

TEST(sadikov_I_Sum_values_by_columns_matrix_mpi, mpi_pipline_reference_run) {
  boost::mpi::communicator world;
  const int columns = 4000;
  const int rows = 4000;
  std::vector<int> in;
  std::vector<int> in_index{rows, columns};
  std::vector<int> out_par(columns, 0);
  std::vector<int> answer(columns, columns);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = std::vector<int>(rows * columns, 1);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in_index[0]);
    taskData->inputs_count.emplace_back(in_index[1]);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskData->outputs_count.emplace_back(out_par.size());
  }
  auto sv_par = std::make_shared<sadikov_I_gather_mpi::ReferenceTask>(taskData);
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

TEST(sadikov_I_Sum_values_by_columns_matrix_mpi, mpi_task_run) {
  boost::mpi::communicator world;
  const int columns = 4000;
  const int rows = 4000;
  const int root = 0;
  std::vector<int> in(columns * rows, 1);
  std::vector<int> in_index{rows, columns};
  std::vector<int> out_par(columns, 0);
  std::vector<int> answer(columns, columns);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = std::vector<int>(rows * columns, 1);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in_index[0]);
    taskData->inputs_count.emplace_back(in_index[1]);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskData->outputs_count.emplace_back(out_par.size());
  }
  auto sv_par = std::make_shared<sadikov_I_gather_mpi::MPITaskParallel>(taskData);
  sv_par->SetRoot(root);
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
  if (world.rank() == root) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(answer, out_par);
  }
}

TEST(sadikov_I_Sum_values_by_columns_matrix_mpi, mpi_reference_run) {
  boost::mpi::communicator world;
  const int columns = 4000;
  const int rows = 4000;
  std::vector<int> in(columns * rows, 1);
  std::vector<int> in_index{rows, columns};
  std::vector<int> out_par(columns, 0);
  std::vector<int> answer(columns, columns);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = std::vector<int>(rows * columns, 1);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in_index[0]);
    taskData->inputs_count.emplace_back(in_index[1]);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskData->outputs_count.emplace_back(out_par.size());
  }
  auto sv_par = std::make_shared<sadikov_I_gather_mpi::ReferenceTask>(taskData);
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