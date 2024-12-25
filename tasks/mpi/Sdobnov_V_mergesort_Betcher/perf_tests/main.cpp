#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/Sdobnov_V_mergesort_Betcher/include/ops_mpi.hpp"

TEST(Sdobnov_V_mergesort_Betcher_par, test_pipeline_run) {
  boost::mpi::communicator world;

  int size = 4096;
  std::vector<int> res(size, 0);
  std::vector<int> input = Sdobnov_V_mergesort_Betcher_par::generate_random_vector(size, 0, 1000);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  }

  auto test = std::make_shared<Sdobnov_V_mergesort_Betcher_par::MergesortBetcherPar>(taskDataPar);
  test->validation();
  test->pre_processing();
  test->run();
  test->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  if (world.rank() == 0) {
    ASSERT_EQ(size, res.size());
  }
}

TEST(Sdobnov_V_mergesort_Betcher_par, test_task_run) {
  boost::mpi::communicator world;

  int size = 4096;
  std::vector<int> res(size, 0);
  std::vector<int> input = Sdobnov_V_mergesort_Betcher_par::generate_random_vector(size, 0, 1000);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  }

  auto test = std::make_shared<Sdobnov_V_mergesort_Betcher_par::MergesortBetcherPar>(taskDataPar);
  test->validation(), true;
  test->pre_processing();
  test->run();
  test->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  if (world.rank() == 0) {
    ASSERT_EQ(size, res.size());
  }
}