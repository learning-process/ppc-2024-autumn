#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/timer.hpp>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/budazhapova_odd_even_merge/include/odd_even_merge.hpp"

namespace budazhapova_betcher_odd_even_merge_mpi {
std::vector<int> generate_random_vector(int size, int minValue, int maxValue) {
  std::vector<int> randomVector(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(minValue, maxValue);
  for (int i = 0; i < size; ++i) {
    randomVector[i] = dis(gen);
  }

  return randomVector;
}
}  // namespace budazhapova_betcher_odd_even_merge_mpi

TEST(budazhapova_betcher_odd_even_merge_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> input_vector = budazhapova_betcher_odd_even_merge_mpi::generate_random_vector(10000000, 5, 100);
  std::vector<int> out(10000000, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    taskDataPar->inputs_count.emplace_back(input_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel = std::make_shared<budazhapova_betcher_odd_even_merge_mpi::MergeParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
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
  }
}

TEST(budazhapova_betcher_odd_even_merge_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> input_vector = budazhapova_betcher_odd_even_merge_mpi::generate_random_vector(10000000, 5, 100);
  std::vector<int> out(10000000, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    taskDataPar->inputs_count.emplace_back(input_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel = std::make_shared<budazhapova_betcher_odd_even_merge_mpi::MergeParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
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
  }
}
