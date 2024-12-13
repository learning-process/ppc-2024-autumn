// Copyright 2024 Korobeinikov Arseny
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/korobeinikov_matrix_multiplication_horizontal_scheme_A_vertical_scheme_B/include/ops_mpi_korobeinikov.hpp"

// mpiexec -n 4 mpi_perf_tests

TEST(mpi_korobeinikov_perf_test_lab_02, test_pipeline_run) {
  boost::mpi::communicator world;

  // Create data
  std::vector<int> A;
  std::vector<int> B;
  int count_rows_A = 150;
  int count_cols_A = 150;
  int count_rows_B = 150;
  int count_cols_B = 150;

  std::vector<int> out;
  int count_rows_out = 0;
  int count_cols_out = 0;
  int count_rows_RA = 150;
  int count_cols_RA = 150;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    A = std::vector<int>(22500, 1);
    B = std::vector<int>(22500, 1);
    out = std::vector<int>(22500, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_A));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_A));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(B.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_B));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_B));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_out));
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_out));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testMpiTaskParallel = std::make_shared<korobeinikov_a_test_task_mpi_lab_02::TestMPITaskParallel>(taskDataPar);
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
    for (size_t i = 0; i < out.size(); i++) {
      ASSERT_EQ(150, out[i]);
    }
    ASSERT_EQ(count_rows_out, count_rows_RA);
    ASSERT_EQ(count_cols_out, count_cols_RA);
  }
}

TEST(mpi_korobeinikov_perf_test_lab_02, test_task_run) {
  boost::mpi::communicator world;

  // Create data
  std::vector<int> A;
  std::vector<int> B;
  int count_rows_A = 150;
  int count_cols_A = 150;
  int count_rows_B = 150;
  int count_cols_B = 150;

  std::vector<int> out;
  int count_rows_out = 0;
  int count_cols_out = 0;
  int count_rows_RA = 150;
  int count_cols_RA = 150;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    A = std::vector<int>(22500, 1);
    B = std::vector<int>(22500, 1);
    out = std::vector<int>(22500, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_A));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_A));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(B.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_B));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_B));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_out));
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_out));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testMpiTaskParallel = std::make_shared<korobeinikov_a_test_task_mpi_lab_02::TestMPITaskParallel>(taskDataPar);
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
    for (size_t i = 0; i < out.size(); i++) {
      ASSERT_EQ(150, out[i]);
    }
    ASSERT_EQ(count_rows_out, count_rows_RA);
    ASSERT_EQ(count_cols_out, count_cols_RA);
  }
}
