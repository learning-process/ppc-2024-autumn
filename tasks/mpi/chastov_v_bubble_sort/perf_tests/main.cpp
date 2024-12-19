#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <cmath>
#include <ctime>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/chastov_v_bubble_sort/include/ops_mpi.hpp"

TEST(chastov_v_bubble_sort, test_pipeline_run) {
  boost::mpi::communicator world;
  int rank = world.rank();
  std::vector<int> global_vec;
  std::vector<int> global_out;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  const size_t count_size_vector = 500;
  if (rank == 0) {
    global_vec = std::vector<int>(count_size_vector, 1);
    global_out = std::vector<int>(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }

  auto testMpiParallel = std::make_shared<chastov_v_bubble_sort::TestMPITaskParallel<int>>(taskDataPar);
  ASSERT_TRUE(testMpiParallel->validation());
  testMpiParallel->pre_processing();
  testMpiParallel->run();
  testMpiParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (rank == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    int count = 0;
    for (size_t i = 0; i < count_size_vector; i++) {
      if (global_vec[i] != global_out[i]) count++;
    }
    ASSERT_EQ(count, 0);
  }
}

TEST(chastov_v_bubble_sort, test_task_run) {
  boost::mpi::communicator world;
  int rank = world.rank();
  std::vector<double> global_vec;
  std::vector<double> global_out;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  const size_t count_size_vector = 10000;
  if (rank == 0) {
    global_vec = std::vector<double>(count_size_vector);
    global_out = std::vector<double>(count_size_vector);
    auto max = static_cast<double>(1000000);
    auto min = static_cast<double>(-1000000);
    std::srand(std::time(nullptr));
    for (size_t i = 0; i < count_size_vector; i++) {
      global_vec[i] = min + static_cast<double>(rand()) / RAND_MAX * (max - min);
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }

  auto testMpiParallel = std::make_shared<chastov_v_bubble_sort::TestMPITaskParallel<double>>(taskDataPar);
  ASSERT_TRUE(testMpiParallel->validation());
  testMpiParallel->pre_processing();
  testMpiParallel->run();
  testMpiParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (rank == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    std::sort(global_vec.begin(), global_vec.end(), [](double a, double b) { return a < b; });
    int count = 0;
    for (size_t i = 0; i < count_size_vector; i++) {
      if (global_vec[i] != global_out[i]) count++;
    }
    ASSERT_EQ(count, 0);
  }
}