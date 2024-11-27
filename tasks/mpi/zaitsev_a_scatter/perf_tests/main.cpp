// Copyright 2023 Liolya Seledkina
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/zaitsev_a_scatter/include/ops_mpi.hpp"

TEST(zaitsev_a_scatter, test_pipeline_run__func_handwritten) {
  boost::mpi::communicator world;
  std::vector<int> inp_vector;
  int extrema = -1;
  int root = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int sz;
  if (world.rank() == root) {
    sz = 120;
    inp_vector = std::vector<int>(sz, 1);
    inp_vector[0] = extrema;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inp_vector.data()));
    taskDataPar->inputs_count.emplace_back(inp_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(inp_vector.data()));
    taskDataPar->outputs_count.emplace_back(inp_vector.size());
  }

  auto testMpiTaskParallel = std::make_shared<zaitsev_a_scatter::TestMPITaskParallel<int, zaitsev_a_scatter::scatter>>(
      taskDataPar, root, MPI_INT);
  if (!testMpiTaskParallel->validation()) {
    GTEST_SKIP();
    return;
  }

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
    ASSERT_EQ(extrema, reinterpret_cast<int*>(taskDataPar->outputs[0])[0]);
  }
}

TEST(zaitsev_a_scatter, test_task_run__func_handwritten) {
  boost::mpi::communicator world;
  std::vector<int> inp_vector;
  int sz;
  int extrema = -1;
  int root = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == root) {
    sz = 120;
    inp_vector = std::vector<int>(sz, 1);
    inp_vector[sz / 3] = extrema;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inp_vector.data()));
    taskDataPar->inputs_count.emplace_back(inp_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(inp_vector.data()));
    taskDataPar->outputs_count.emplace_back(inp_vector.size());
  }

  auto testMpiTaskParallel = std::make_shared<zaitsev_a_scatter::TestMPITaskParallel<int, zaitsev_a_scatter::scatter>>(
      taskDataPar, root, MPI_INT);
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
  if (world.rank() == root) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(extrema, reinterpret_cast<int*>(taskDataPar->outputs[0])[0]);
  }
}

TEST(zaitsev_a_scatter, test_pipeline_run__func_builtin) {
  boost::mpi::communicator world;
  std::vector<int> inp_vector;
  int extrema = -1;
  int root = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int sz;
  if (world.rank() == root) {
    sz = 120;
    inp_vector = std::vector<int>(sz, 1);
    inp_vector[0] = extrema;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inp_vector.data()));
    taskDataPar->inputs_count.emplace_back(inp_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(inp_vector.data()));
    taskDataPar->outputs_count.emplace_back(inp_vector.size());
  }

  auto testMpiTaskParallel = std::make_shared<zaitsev_a_scatter::TestMPITaskParallel<int, MPI_Scatter>>(
      taskDataPar, root, MPI_INT);
  if (!testMpiTaskParallel->validation()) {
    GTEST_SKIP();
    return;
  }

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
    ASSERT_EQ(extrema, reinterpret_cast<int*>(taskDataPar->outputs[0])[0]);
  }
}

TEST(zaitsev_a_scatter, test_task_run__func_builtin) {
  boost::mpi::communicator world;
  std::vector<int> inp_vector;
  int sz;
  int extrema = -1;
  int root = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == root) {
    sz = 120;
    inp_vector = std::vector<int>(sz, 1);
    inp_vector[sz / 3] = extrema;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inp_vector.data()));
    taskDataPar->inputs_count.emplace_back(inp_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(inp_vector.data()));
    taskDataPar->outputs_count.emplace_back(inp_vector.size());
  }

  auto testMpiTaskParallel = std::make_shared<zaitsev_a_scatter::TestMPITaskParallel<int, MPI_Scatter>>(
      taskDataPar, root, MPI_INT);
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
  if (world.rank() == root) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(extrema, reinterpret_cast<int*>(taskDataPar->outputs[0])[0]);
  }
}
