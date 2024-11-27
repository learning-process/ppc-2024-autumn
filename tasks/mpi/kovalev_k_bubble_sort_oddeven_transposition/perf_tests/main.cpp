#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <cmath>
#include <ctime>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kovalev_k_bubble_sort_oddeven_transposition/include/header.hpp"

TEST(kovalev_k_bubble_sort_oddeven_transposition_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int rank = world.rank();
  std::vector<int> g_vec;
  std::vector<int> g_out;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  const size_t length = 300;
  const int alpha = 1;
  if (rank == 0) {
    g_vec = std::vector<int>(length, alpha);
    g_out = std::vector<int>(length);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(g_vec.data()));
    taskDataPar->inputs_count.emplace_back(g_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(g_out.data()));
    taskDataPar->outputs_count.emplace_back(g_out.size());
  }
  auto testMpiParallel =
      std::make_shared<kovalev_k_bubble_sort_oddeven_transposition_mpi::BubbleSortOddEvenTranspositionPar<int>>(
          taskDataPar);
  ASSERT_TRUE(testMpiParallel->validation());
  testMpiParallel->pre_processing();
  testMpiParallel->run();
  testMpiParallel->post_processing();
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (rank == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    int count_viol = 0;
    for (int i = 0; i < length; i++) {
      if (g_vec[i] != g_out[i]) count_viol++;
    }
    ASSERT_EQ(count_viol, 0);
  }
}

TEST(kovalev_k_bubble_sort_oddeven_transposition_mpi, test_task_run) {
  boost::mpi::communicator world;
  int rank = world.rank();
  std::vector<double> g_vec;
  std::vector<double> g_out;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  const size_t length = 30000;
  if (rank == 0) {
    g_vec = std::vector<double>(length);
    g_out = std::vector<double>(length);
    auto max = static_cast<double>(1000000);
    auto min = static_cast<double>(-1000000);
    std::srand(std::time(nullptr));
    for (size_t i = 0; i < length; i++) g_vec[i] = min + static_cast<double>(rand()) / RAND_MAX * (max - min);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(g_vec.data()));
    taskDataPar->inputs_count.emplace_back(g_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(g_out.data()));
    taskDataPar->outputs_count.emplace_back(g_out.size());
  }
  auto testMpiParallel =
      std::make_shared<kovalev_k_bubble_sort_oddeven_transposition_mpi::BubbleSortOddEvenTranspositionPar<double>>(taskDataPar);
  ASSERT_TRUE(testMpiParallel->validation());
  testMpiParallel->pre_processing();
  testMpiParallel->run();
  testMpiParallel->post_processing();
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (rank == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    std::sort(g_vec.begin(), g_vec.end(), [](double a, double b) { return a < b; });
    int count_viol = 0;
    for (int i = 0; i < length; i++) {
      if (g_vec[i] != g_out[i]) count_viol++;
    }
    ASSERT_EQ(count_viol, 0);
  }
}
