#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/tyshkevich_a_num_of_orderly_violations/include/ops_mpi.hpp"

TEST(tyshkevich_a_num_of_orderly_violations_mpi_ptest, test_pipeline_run) {
  int size = 999;

  // Create data
  std::vector<int> global_vec(size);
  std::vector<int> result(1, 0);

  boost::mpi::communicator world;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = tyshkevich_a_num_of_orderly_violations_mpi::getRandomVector(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }

  // Create Task
  auto testMpiTaskParallel =
      std::make_shared<tyshkevich_a_num_of_orderly_violations_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ((uint32_t)(1), taskDataPar->outputs_count[0]);
  }
}

TEST(tyshkevich_a_num_of_orderly_violations_mpi_ptest, test_task_run) {
  int size = 999;

  // Create data
  std::vector<int> global_vec(size);
  std::vector<int> result(1, 0);

  boost::mpi::communicator world;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = tyshkevich_a_num_of_orderly_violations_mpi::getRandomVector(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }

  // Create Task
  auto testMpiTaskParallel =
      std::make_shared<tyshkevich_a_num_of_orderly_violations_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ((uint32_t)(1), taskDataPar->outputs_count[0]);
  }
}