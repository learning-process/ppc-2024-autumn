// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/frolova_e_Simpson_method/include/ops_mpi_frolova_Simpson.hpp"

TEST(frolova_e_Simpson_method_mpi, test_pipeline_run) {
  // Create data
  boost::mpi::communicator world;
  std::vector<int> values_1 = {1000, 2, 4};
  std::vector<int> values_11 = {1000, 2};
  std::vector<double> values_2 = {0.0, 1000.0, 0.0, 1000.0};

  std::vector<double> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
    taskDataPar->inputs_count.emplace_back(values_1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
    taskDataPar->inputs_count.emplace_back(values_2.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size() * sizeof(double));
  }

  auto testMpiTaskParallel = std::make_shared<frolova_e_Simpson_method_mpi::SimpsonmethodParallel>(taskDataPar);
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
    // Create data
    ppc::core::Perf::print_perf_statistic(perfResults);

    std::vector<double> res_2(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_11.data()));
    taskDataSeq->inputs_count.emplace_back(values_11.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
    taskDataSeq->inputs_count.emplace_back(values_2.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_2.data()));
    taskDataSeq->outputs_count.emplace_back(res_2.size() * sizeof(double));

    // Create Task
    frolova_e_Simpson_method_mpi::SimpsonmethodSequential testTaskSequential(
        taskDataSeq, frolova_e_Simpson_method_mpi::ProductOfXAndY);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_NEAR(res_2[0], res[0], 0.1);
  }
}

TEST(frolova_e_Simpson_method_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> values_1 = {1000, 2, 4};
  std::vector<int> values_11 = {1000, 2};
  std::vector<double> values_2 = {0.0, 1000.0, 0.0, 1000.0};

  std::vector<double> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
    taskDataPar->inputs_count.emplace_back(values_1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
    taskDataPar->inputs_count.emplace_back(values_2.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size() * sizeof(double));
  }

  auto testMpiTaskParallel = std::make_shared<frolova_e_Simpson_method_mpi::SimpsonmethodParallel>(taskDataPar);
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
    std::vector<double> res_2(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_11.data()));
    taskDataSeq->inputs_count.emplace_back(values_11.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
    taskDataSeq->inputs_count.emplace_back(values_2.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_2.data()));
    taskDataSeq->outputs_count.emplace_back(res_2.size() * sizeof(double));

    // Create Task
    frolova_e_Simpson_method_mpi::SimpsonmethodSequential testMpiTaskSequential(
        taskDataSeq, frolova_e_Simpson_method_mpi::ProductOfXAndY);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_NEAR(res_2[0], res[0], 0.1);
  }
}
