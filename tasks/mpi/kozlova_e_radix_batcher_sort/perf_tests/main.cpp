// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kozlova_e_radix_batcher_sort/include/ops_mpi.hpp"

namespace kozlova_e_utility_functions {
std::vector<double> generate_random_double_vector(size_t size, double min_val, double max_val) {
  std::vector<double> result(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(min_val, max_val);

  for (auto& value : result) {
    value = dist(gen);
  }
  return result;
}
}  // namespace kozlova_e_utility_functions

TEST(kozlova_e_radix_batcher_sort_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  const int size = 100000;
  std::vector<double> global_vec;
  std::vector<double> resMPI(size, 0);
  std::vector<double> resSeq;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec = kozlova_e_utility_functions::generate_random_double_vector(size, -100.0, 100.0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  auto testMpiTaskParallel = std::make_shared<kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortMPI>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    resSeq.resize(size);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(resSeq.size());

    kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortSequential testMPITaskSequential(taskDataSeq);

    ASSERT_EQ(testMPITaskSequential.validation(), true);
    testMPITaskSequential.pre_processing();
    testMPITaskSequential.run();
    testMPITaskSequential.post_processing();
  }

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
    ASSERT_EQ(resSeq, resMPI);
  }
}

TEST(kozlova_e_radix_batcher_sort_mpi, test_task_run) {
  boost::mpi::communicator world;
  const int size = 100000;
  std::vector<double> global_vec;
  std::vector<double> resMPI(size, 0);
  std::vector<double> resSeq;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec = kozlova_e_utility_functions::generate_random_double_vector(size, -100.0, 100.0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resMPI.data()));
    taskDataPar->outputs_count.emplace_back(resMPI.size());
  }

  auto testMpiTaskParallel = std::make_shared<kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortMPI>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    resSeq.resize(size);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resSeq.data()));
    taskDataSeq->outputs_count.emplace_back(resSeq.size());

    kozlova_e_radix_batcher_sort_mpi::RadixBatcherSortSequential testMPITaskSequential(taskDataSeq);

    ASSERT_EQ(testMPITaskSequential.validation(), true);
    testMPITaskSequential.pre_processing();
    testMPITaskSequential.run();
    testMPITaskSequential.post_processing();
  }

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
    ASSERT_EQ(resSeq, resMPI);
  }
}
