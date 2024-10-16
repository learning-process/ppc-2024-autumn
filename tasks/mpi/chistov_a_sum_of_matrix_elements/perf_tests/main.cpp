#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/chistov_a_sum_of_matrix_elements/include/ops_mpi.hpp"

TEST(chistov_a_sum_of_matrix_elements, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> global_sum(1, 0);
  const int n = 10000;
  const int m = 10000;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_matrix = chistov_a_sum_of_matrix_elements::getRandomMatrix<int>(n, m);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<chistov_a_sum_of_matrix_elements::TestMPITaskParallel<int>>(taskDataPar, n, m);

  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  ASSERT_EQ(testMpiTaskParallel->pre_processing(), true);
  ASSERT_EQ(testMpiTaskParallel->run(), true);
  ASSERT_EQ(testMpiTaskParallel->post_processing(), true);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(std::accumulate(global_matrix.begin(), global_matrix.end(), 0), global_sum[0]);
  }
}

TEST(chistov_a_sum_of_matrix_elements, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> global_sum(1, 0);
  const int n = 10000;
  const int m = 10000;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_matrix = std::vector<int>(n * m, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<chistov_a_sum_of_matrix_elements::TestMPITaskParallel<int>>(taskDataPar, n, m);

  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  ASSERT_EQ(testMpiTaskParallel->pre_processing(), true);
  ASSERT_EQ(testMpiTaskParallel->run(), true);
  ASSERT_EQ(testMpiTaskParallel->post_processing(), true);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    int expected_sum = std::accumulate(global_matrix.begin(), global_matrix.end(), 0);
    ASSERT_EQ(expected_sum, global_sum[0]);
  }
}

int main(int argc, char** argv) {
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (world.rank() != 0) {
    delete listeners.Release(listeners.default_result_printer());
  }
  return RUN_ALL_TESTS();
}
