#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/beskhmelnova_k_most_different_neighbor_elements/include/mpi.hpp"
#include "mpi/beskhmelnova_k_most_different_neighbor_elements/src/mpi.cpp"

TEST(mpi_beskhmelnova_k_most_different_neighbor_elements_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_out(2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 10000000;
    global_vec = std::vector<int>(count_size_vector, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<beskhmelnova_k_most_different_neighbor_elements_mpi::TestMPITaskParallel<int>>(taskDataPar);
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
    int index = beskhmelnova_k_most_different_neighbor_elements_mpi::position_of_first_neighbour_seq(global_vec);
    ASSERT_EQ(global_vec[index], global_out[0]);
    ASSERT_EQ(global_vec[index + 1], global_out[1]);
  }
}

TEST(mpi_beskhmelnova_k_most_different_neighbor_elements_perf_test, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_out(2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;
  if (world.rank() == 0) {
    count_size_vector = 25000000;
    global_vec = std::vector<int>(count_size_vector, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<beskhmelnova_k_most_different_neighbor_elements_mpi::TestMPITaskParallel<int>>(taskDataPar);
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
    int index = beskhmelnova_k_most_different_neighbor_elements_mpi::position_of_first_neighbour_seq(global_vec);
    ASSERT_EQ(global_vec[index], global_out[0]);
    ASSERT_EQ(global_vec[index + 1], global_out[1]);
  }
}
<<<<<<< HEAD

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
=======
>>>>>>> upstream/master
