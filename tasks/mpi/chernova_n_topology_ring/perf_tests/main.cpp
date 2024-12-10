#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/chernova_n_topology_ring/include/ops_mpi.hpp"

std::vector<char> generateDataPerf(int k) {
  const std::string words[] = {"one", "two", "three"};
  const int wordArraySize = sizeof(words) / sizeof(words[0]);

  std::string result;

  for (int i = 0; i < k; ++i) {
    result += words[i % wordArraySize];
    if (i < k - 1) {
      result += ' ';
    }
  }

  return std::vector<char>(result.begin(), result.end());
}

TEST(chernova_n_topology_ring_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  const int k = 100000;
  std::vector<char> testDataParallel = generateDataPerf(k);
  std::vector<char> in = testDataParallel;
  const int N = in.size();
  std::vector<char> out_vec(N);
  std::vector<int> out_process;

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    out_process = std::vector<int>(world.size() + 1);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(in.data())));
    taskDataParallel->inputs_count.emplace_back(in.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_process.data()));
    taskDataParallel->outputs_count.emplace_back(2);
  }

  auto testMpiTaskParallel = std::make_shared<chernova_n_topology_ring_mpi::TestMPITaskParallel>(taskDataParallel);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(true, std::equal(in.begin(), in.end(), out_vec.begin(), out_vec.end()));
  }
}

TEST(chernova_n_topology_ring_mpi, test_task_run) {
  boost::mpi::communicator world;
  const int k = 100000;
  std::vector<char> testDataParallel = generateDataPerf(k);
  std::vector<char> in = testDataParallel;
  const int N = in.size();
  std::vector<char> out_vec(N);
  std::vector<int> out_process;

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    out_process = std::vector<int>(world.size() + 1);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(in.data())));
    taskDataParallel->inputs_count.emplace_back(in.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_process.data()));
    taskDataParallel->outputs_count.emplace_back(2);
  }

  auto testMpiTaskParallel = std::make_shared<chernova_n_topology_ring_mpi::TestMPITaskParallel>(taskDataParallel);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(true, std::equal(in.begin(), in.end(), out_vec.begin(), out_vec.end()));
  }
}