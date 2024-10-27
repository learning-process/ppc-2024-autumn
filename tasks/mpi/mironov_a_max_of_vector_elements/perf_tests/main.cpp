#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/mironov_a_max_of_vector_elements/include/ops_mpi.hpp"

TEST(mironov_a_max_of_vector_elements_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int gold;
  if (world.rank() == 0) {
    const int count = 20000000;
    const int start = -789000000;
    gold = start + 5 * (count - 1);
    global_vec.resize(count);
    for (int i = 0, j = start; i < count; ++i, j += 5) global_vec[i] = j;

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  auto testMpiTaskParallel = std::make_shared<mironov_a_max_of_vector_elements_mpi::MaxVectorMPI>(taskDataPar);
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
    ASSERT_EQ(gold, global_max[0]);
  }
}

TEST(mironov_a_max_of_vector_elements_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int gold;
  if (world.rank() == 0) {
    const int count = 200000000;
    const int start = -789000000;
    gold = start + 5 * (count - 1);
    global_vec.resize(count);
    for (int i = 0, j = start; i < count; ++i, j += 5) global_vec[i] = j;

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  auto testMpiTaskParallel = std::make_shared<mironov_a_max_of_vector_elements_mpi::MaxVectorMPI>(taskDataPar);
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
    ASSERT_EQ(gold, global_max[0]);
  }
}
