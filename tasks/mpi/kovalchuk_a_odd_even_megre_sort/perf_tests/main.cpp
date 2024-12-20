#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kovalchuk_a_odd_even_megre_sort/include/ops_mpi.hpp"

TEST(mpi_sort_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_sorted_vec;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 10000;
    global_vec = std::vector<int>(count_size_vector);
    std::generate(global_vec.begin(), global_vec.end(), std::rand);
    global_sorted_vec = std::vector<int>(count_size_vector);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sorted_vec.data()));
    taskDataPar->outputs_count.emplace_back(global_sorted_vec.size());
  }

  auto testMpiTaskParallel = std::make_shared<kovalchuk_a_odd_even::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;  // Run 10 iterations for averaging
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_TRUE(std::is_sorted(global_sorted_vec.begin(), global_sorted_vec.end()));
  }
}

TEST(mpi_sort_perf_test, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_sorted_vec;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 10000;
    global_vec = std::vector<int>(count_size_vector);
    std::generate(global_vec.begin(), global_vec.end(), std::rand);
    global_sorted_vec = std::vector<int>(count_size_vector);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sorted_vec.data()));
    taskDataPar->outputs_count.emplace_back(global_sorted_vec.size());
  }

  auto testMpiTaskParallel = std::make_shared<kovalchuk_a_odd_even::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_TRUE(std::is_sorted(global_sorted_vec.begin(), global_sorted_vec.end()));
  }
}
