#include "mpi/chistov_a_gather_my/include/sort_my.hpp"

#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"

TEST(chistov_a_gather_my, test_pipeline_run) {
  boost::mpi::communicator world;
  const int count_size_vector = 10000000;
  std::vector<int> vector;
  std::vector<int> local_vec = chistov_a_gather_my::getRandomVector<int>(count_size_vector);
  std::vector<int> gathered_data(count_size_vector * world.size());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(local_vec.data()));
  taskDataPar->inputs_count.emplace_back(count_size_vector);

  if (world.rank() == 0) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(gathered_data.data()));
    taskDataPar->outputs_count.emplace_back(gathered_data.size());
  }

  chistov_a_gather_my::Sorting<int> testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer =
      std::make_shared<ppc::core::Perf>(std::make_shared<chistov_a_gather_my::Sorting<int>>(taskDataPar));
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  boost::mpi::gather(world, local_vec.data(), count_size_vector, vector, 0);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    std::sort(vector.begin(), vector.end());
    ASSERT_EQ(gathered_data, vector);
  }
}

TEST(chistov_a_gather_my, test_task_run) {
  boost::mpi::communicator world;
  const int count_size_vector = 10000000;
  std::vector<int> vector;
  std::vector<int> local_vec = chistov_a_gather_my::getRandomVector<int>(count_size_vector);
  std::vector<int> gathered_data(count_size_vector * world.size());

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(local_vec.data()));
  taskDataPar->inputs_count.emplace_back(count_size_vector);

  if (world.rank() == 0) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(gathered_data.data()));
    taskDataPar->outputs_count.emplace_back(gathered_data.size());
  }

  chistov_a_gather_my::Sorting<int> testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer =
      std::make_shared<ppc::core::Perf>(std::make_shared<chistov_a_gather_my::Sorting<int>>(taskDataPar));
  perfAnalyzer->task_run(perfAttr, perfResults);

  boost::mpi::gather(world, local_vec.data(), count_size_vector, vector, 0);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    std::sort(vector.begin(), vector.end());
    ASSERT_EQ(gathered_data, vector);
  }
}