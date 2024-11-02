// Copyright 2024 Alputov Ivan
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/alputov_i_most_diff_neighb_elem/include/ops_mpi.hpp"
#include "mpi/alputov_i_most_diff_neighb_elem/src/ops_mpi.cpp"

TEST(mpi_alputov_i_most_diff_neighb_elem_mpi_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> inputVector;
  int outputPair[2];

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count = 125000000;

  if (world.rank() == 0) {
    inputVector = std::vector<int>(count);
    for (size_t i = 0; i < inputVector.size(); i++) {
      inputVector[i] = i;
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputVector.data()));
    taskDataPar->inputs_count.emplace_back(inputVector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputPair));
    taskDataPar->outputs_count.emplace_back(2);
  }

  auto testMpiTaskParallel = std::make_shared<alputov_i_most_diff_neighb_elem_mpi::MPIParallelTask>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  bool pre_processing_result = testMpiTaskParallel->pre_processing();
  ASSERT_TRUE(pre_processing_result);
  bool run_result = testMpiTaskParallel->run();
  ASSERT_TRUE(run_result);
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
    int expectedMaxDifferenceindex = alputov_i_most_diff_neighb_elem_mpi::Max_Neighbour_Seq_Pos(inputVector);
    ASSERT_EQ(std::abs(outputPair[1] - outputPair[0]),
              std::abs(inputVector[expectedMaxDifferenceindex + 1] - inputVector[expectedMaxDifferenceindex]));
  }
}

TEST(mpi_alputov_i_most_diff_neighb_elem_mpi_perf_test, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> inputVector;
  int32_t outputPair[2];

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count = 125000000;
  if (world.rank() == 0) {
    inputVector = std::vector<int>(count);
    for (size_t i = 0; i < inputVector.size(); i++) {
      inputVector[i] = i;
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputVector.data()));
    taskDataPar->inputs_count.emplace_back(inputVector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputPair));
    taskDataPar->outputs_count.emplace_back(2);
  }

  auto testMpiTaskParallel = std::make_shared<alputov_i_most_diff_neighb_elem_mpi::MPIParallelTask>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  bool pre_processing_result = testMpiTaskParallel->pre_processing();
  ASSERT_TRUE(pre_processing_result);
  bool run_result = testMpiTaskParallel->run();
  ASSERT_TRUE(run_result);
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
    int expectedMaxDifferenceindex = alputov_i_most_diff_neighb_elem_mpi::Max_Neighbour_Seq_Pos(inputVector);
    ASSERT_EQ(std::abs(outputPair[1] - outputPair[0]),
              std::abs(inputVector[expectedMaxDifferenceindex + 1] - inputVector[expectedMaxDifferenceindex]));
  }
}