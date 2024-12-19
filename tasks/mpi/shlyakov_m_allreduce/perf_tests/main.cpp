// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/shlyakov_m_allreduce/include/ops_mpi.hpp"

TEST(shlyakov_m_all_reduce_mpi, test_pipeline_run) {
  int row = 1000;
  int col = 4000;
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> m_vec(row);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = std::vector<int>(row * col, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(m_vec.size());
  }

  auto testMpiTaskParallel = std::make_shared<shlyakov_m_all_reduce_mpi::TestMPITaskParallel>(taskDataPar);
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
    for (unsigned i = 0; i < m_vec.size(); i++) {
      EXPECT_EQ(4000, m_vec[i]);
    }
  }
}

TEST(shlyakov_m_all_reduce_mpi, test_task_run) {
  int row = 1000;
  int col = 4000;
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> m_vec(row);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = std::vector<int>(row * col, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(m_vec.size());
  }

  auto testMpiTaskParallel = std::make_shared<shlyakov_m_all_reduce_mpi::TestMPITaskParallel>(taskDataPar);
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
    for (unsigned i = 0; i < m_vec.size(); i++) {
      EXPECT_EQ(4000, m_vec[i]);
    }
  }
}

TEST(shlyakov_m_all_reduce_mpi, My_test_pipeline_run) {
  int row = 1000;
  int col = 4000;
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> m_vec(row);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = std::vector<int>(row * col, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(m_vec.size());
  }

  auto testMpiTaskParallel = std::make_shared<shlyakov_m_all_reduce_mpi::MyTestMPITaskParallel>(taskDataPar);
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
    for (unsigned i = 0; i < m_vec.size(); i++) {
      EXPECT_EQ(4000, m_vec[i]);
    }
  }
}

TEST(shlyakov_m_all_reduce_mpi, My_test_task_run) {
  int row = 1000;
  int col = 4000;
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> m_vec(row);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = std::vector<int>(row * col, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(m_vec.size());
  }

  auto testMpiTaskParallel = std::make_shared<shlyakov_m_all_reduce_mpi::MyTestMPITaskParallel>(taskDataPar);
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
    for (unsigned i = 0; i < m_vec.size(); i++) {
      EXPECT_EQ(4000, m_vec[i]);
    }
  }
}