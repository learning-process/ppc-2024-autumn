#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/laganina_e_readers_writers/include/ops_mpi.hpp"

namespace laganina_e_readers_writers_perf_tests {

std::vector<int> getRandomVector(int sz, int a, int b) {
  std::random_device dev;
  std::mt19937 gen(dev());
  if (a > b) {
    throw std::out_of_range("range is incorrect");
  }
  std::uniform_int_distribution<> dist(a, b);

  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}
}  // namespace laganina_e_readers_writers_perf_tests

TEST(laganina_e_readers_writers_mpi_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;

  if (world.size() < 2) {
    GTEST_SKIP();
  }

  const int count_size_vector = 500;
  int count = 0;  // this variable is needed to find out how many writers there were
  std::vector<int> global_vec;
  ASSERT_NO_THROW(global_vec = laganina_e_readers_writers_perf_tests::getRandomVector(count_size_vector, -1000, 1000));

  std::vector<int> out_vec(count_size_vector, 0);
  std::vector<int> exp_parallel = global_vec;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_vec.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&count));
    taskDataPar->outputs_count.emplace_back(out_vec.size());
  }

  auto testMpiTaskParallel = std::make_shared<laganina_e_readers_writers_mpi::TestMPITaskParallel>(taskDataPar);
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
    for (int i = 0; i < count_size_vector; i++) {
      exp_parallel[i] += count;
    }
    ASSERT_EQ(exp_parallel, out_vec);
  }
}

TEST(laganina_e_readers_writers_mpi_perf_test, test_task_run) {
  boost::mpi::communicator world;

  if (world.size() < 2) {
    GTEST_SKIP();
  }

  const int count_size_vector = 500;
  int count = 0;  // this variable is needed to find out how many writers there were
  std::vector<int> global_vec;
  ASSERT_NO_THROW(global_vec = laganina_e_readers_writers_perf_tests::getRandomVector(count_size_vector, -1000, 1000));

  std::vector<int> out_vec(count_size_vector, 0);
  std::vector<int> exp_parallel = global_vec;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_vec.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&count));
    taskDataPar->outputs_count.emplace_back(out_vec.size());
  }

  auto testMpiTaskParallel = std::make_shared<laganina_e_readers_writers_mpi::TestMPITaskParallel>(taskDataPar);
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
    for (int i = 0; i < count_size_vector; i++) {
      exp_parallel[i] += count;
    }
    ASSERT_EQ(exp_parallel, out_vec);
  }
}
