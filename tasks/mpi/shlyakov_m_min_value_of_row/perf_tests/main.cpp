// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/shlyakov_m_min_value_of_row/include/ops_mpi.hpp"

TEST(shlyakov_m_min_value_of_row_mpi, test_pipeline_run_min) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> main_matr;
  std::vector<int32_t> main_min;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int sz_row;
  int sz_col;

  if (world.rank() == 0) {
    sz_row = 5000;
    sz_col = 5000;
    main_matr = shlyakov_m_min_value_of_row_mpi::TestMPITaskSequential::get_random_matr(sz_row, sz_col);
    main_min.resize(sz_row, INT_MAX);

    for (auto& row : main_matr) taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));

    taskDataPar->inputs_count.emplace_back(sz_row);
    taskDataPar->inputs_count.emplace_back(sz_col);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(main_min.data()));
    taskDataPar->outputs_count.emplace_back(main_min.size());
  }

  auto testMpiTaskParallel = std::make_shared<shlyakov_m_min_value_of_row_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    for (size_t i = 0; i < main_min.size(); ++i) ASSERT_EQ(main_min[i], INT_MIN);
  }
}

TEST(shlyakov_m_min_value_of_row_mpi_perf, test_task_run_min) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> main_matr;
  std::vector<int32_t> main_min;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int sz_row;
  int sz_col;

  if (world.rank() == 0) {
    sz_row = 5000;
    sz_col = 5000;
    main_matr = shlyakov_m_min_value_of_row_mpi::TestMPITaskSequential::get_random_matr(sz_row, sz_col);
    main_min.resize(sz_row, INT_MAX);

    for (auto& row : main_matr) taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));

    taskDataPar->inputs_count.emplace_back(sz_row);
    taskDataPar->inputs_count.emplace_back(sz_col);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(main_min.data()));
    taskDataPar->outputs_count.emplace_back(main_min.size());
  }

  auto testMpiTaskParallel = std::make_shared<shlyakov_m_min_value_of_row_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    for (size_t i = 0; i < main_min.size(); ++i) ASSERT_EQ(main_min[i], INT_MIN);
  }
}