// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/tsatsyn_a_vector_dot_product/include/ops_mpi.hpp"
std::vector<int> GetRandomVector(int size) {
  std::vector<int> vector(size);
  std::srand((time(NULL)));
  for (int i = 0; i < size; ++i) vector[i] = std::rand() % 100 + 1;

  return vector;
}
TEST(mpi_tsatsyn_a_vector_dot_product_perf_test, test_pipeline_run) {
  int size = 10000000;
  boost::mpi::communicator world;
  std::vector<int> v1 = GetRandomVector(size);
  std::vector<int> v2 = GetRandomVector(size);
  std::vector<int> ans(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
    taskDataPar->inputs_count.emplace_back(v1.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
    taskDataPar->inputs_count.emplace_back(v2.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans.data()));
    taskDataPar->outputs_count.emplace_back(ans.size());
  }

  auto testMpiTaskParallel = std::make_shared<tsatsyn_a_vector_dot_product_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(tsatsyn_a_vector_dot_product_mpi::resulting(v1, v2), ans[0]);
  }
}

TEST(mpi_tsatsyn_a_vector_dot_product_perf_test, test_task_run) {
  int size = 10000000;
  boost::mpi::communicator world;
  std::vector<int> v1 = GetRandomVector(size);
  std::vector<int> v2 = GetRandomVector(size);
  std::vector<int> ans(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
    taskDataPar->inputs_count.emplace_back(v1.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
    taskDataPar->inputs_count.emplace_back(v2.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans.data()));
    taskDataPar->outputs_count.emplace_back(ans.size());
  }

  auto testMpiTaskParallel = std::make_shared<tsatsyn_a_vector_dot_product_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(tsatsyn_a_vector_dot_product_mpi::resulting(v1, v2), ans[0]);
  }
}

// int main(int argc, char** argv) {
//   boost::mpi::environment env(argc, argv);
//   boost::mpi::communicator world;
//   ::testing::InitGoogleTest(&argc, argv);
//   ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
//   if (world.rank() != 0) {
//     delete listeners.Release(listeners.default_result_printer());
//   }
//   return RUN_ALL_TESTS();
// }
