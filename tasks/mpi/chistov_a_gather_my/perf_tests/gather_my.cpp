#include "mpi/chistov_a_gather_my/include/gather_my.hpp"

#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"

TEST(chistov_a_gather_my, test_pipeline_run2) {
  boost::mpi::communicator world;
  const int count_size_vector = 100000000;
  std::vector<int> local_vec(count_size_vector, 1);
  std::vector<int> gathered_data;
  std::vector<int> mpi_gathered_data;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  local_vec = std::vector<int>(count_size_vector, 1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(local_vec.data()));
  taskDataPar->inputs_count.emplace_back(count_size_vector);
  if (world.rank() == 0) {
    gathered_data.resize(world.size() * count_size_vector);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(gathered_data.data()));
    taskDataPar->outputs_count.emplace_back(gathered_data.size());
  }
  auto testMpiTaskParallel = std::make_shared<chistov_a_gather_my::Gather<int>>(taskDataPar, 0);

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

  boost::mpi::gather(world, local_vec.data(), count_size_vector, mpi_gathered_data, 0);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_TRUE(std::is_permutation(gathered_data.begin(), gathered_data.end(), mpi_gathered_data.begin()));
  }
}

TEST(chistov_a_gather_my, test_task_run2) {
  boost::mpi::communicator world;
  const int count_size_vector = 100000000;
  std::vector<int> local_vec(count_size_vector, 1);
  std::vector<int> gathered_data;
  std::vector<int> mpi_gathered_data;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  local_vec = std::vector<int>(count_size_vector, 1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(local_vec.data()));
  taskDataPar->inputs_count.emplace_back(count_size_vector);
  if (world.rank() == 0) {
    gathered_data.resize(world.size() * count_size_vector);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(gathered_data.data()));
    taskDataPar->outputs_count.emplace_back(gathered_data.size());
  }
  auto testMpiTaskParallel = std::make_shared<chistov_a_gather_my::Gather<int>>(taskDataPar, 0);

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

  boost::mpi::gather(world, local_vec.data(), count_size_vector, mpi_gathered_data, 0);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_TRUE(std::is_permutation(gathered_data.begin(), gathered_data.end(), mpi_gathered_data.begin()));
  }
}
