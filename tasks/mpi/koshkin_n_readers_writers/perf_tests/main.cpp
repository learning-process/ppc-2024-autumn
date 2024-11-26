#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/koshkin_n_readers_writers/include/ops_mpi.hpp"

TEST(koshkin_n_readers_writers_mpi_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;

  if (world.size() < 2) {
    ASSERT_TRUE(true);
    return;
  }

  const int count_size_vector = 500;

  std::vector<int> global_vec = koshkin_n_readers_writers_mpi::getRandomVector(count_size_vector);
  std::vector<int> exit_vec(count_size_vector, 0);
  std::vector<int> res_exp_paral = global_vec;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int num_writers = 0;
    for (int rank = 1; rank < world.size(); rank++) {
      if (rank % 2 != 0) num_writers++;
    }
    for (int i = 0; i < count_size_vector; i++) {
      res_exp_paral[i] += num_writers * 100;
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(exit_vec.data()));
    taskDataPar->outputs_count.emplace_back(exit_vec.size());
  }

  auto testMpiTaskParallel = std::make_shared<koshkin_n_readers_writers_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(res_exp_paral, exit_vec);
  }
}

TEST(koshkin_n_readers_writers_mpi_perf_test, test_task_run) {
  boost::mpi::communicator world;

  if (world.size() < 2) {
    ASSERT_TRUE(true);
    return;
  }

  const int count_size_vector = 500;

  std::vector<int> global_vec = koshkin_n_readers_writers_mpi::getRandomVector(count_size_vector);
  std::vector<int> exit_vec(count_size_vector, 0);
  std::vector<int> res_exp_paral = global_vec;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    int num_writers = 0;
    for (int rank = 1; rank < world.size(); rank++) {
      if (rank % 2 != 0) num_writers++;
    }
    for (int i = 0; i < count_size_vector; i++) {
      res_exp_paral[i] += num_writers * 100;
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(exit_vec.data()));
    taskDataPar->outputs_count.emplace_back(exit_vec.size());
  }

  auto testMpiTaskParallel = std::make_shared<koshkin_n_readers_writers_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(res_exp_paral, exit_vec);
  }
}