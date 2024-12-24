#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <climits>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/gordeva_t_max_val_of_column_matrix/include/ops_mpi.hpp"

TEST(gordeva_t_max_val_of_column_matrix_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> max_s;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  size_t rows = 5000;
  size_t cols = 5000;

  if (world.rank() == 0) {
    global_matr = gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential::rand_matr(rows, cols);
    max_s.resize(cols, INT_MIN);
    for (auto& i : global_matr) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(i.data()));
    }
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(max_s.data()));
    taskDataPar->outputs_count.emplace_back(max_s.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<gordeva_t_max_val_of_column_matrix_mpi ::TestMPITaskParallel>(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 100;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    std::vector<int> max_example(cols, INT_MIN);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matr.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matr[i].data()));
    }
    taskDataSeq->inputs_count = {rows, cols};
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(max_example.data()));
    taskDataSeq->outputs_count.emplace_back(max_example.size());
    gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(global_max, max_example);
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(gordeva_t_max_val_of_column_matrix_mpi, test_task_run) {
  boost::mpi::communicator world;

  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> max_s;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int rows = 5000;
  int cols = 5000;

  if (world.rank() == 0) {
    global_matr = gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential::rand_matr(rows, cols);
    max_s.resize(cols, INT_MIN);

    for (auto& i : global_matr) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(i.data()));
    }

    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(max_s.data()));
    taskDataPar->outputs_count.emplace_back(max_s.size());
  }

  auto testMpiTaskParallel = std::make_shared<gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskParallel>(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    std::vector<int> max_example(cols, INT_MIN);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < global_matr.size(); i++) {
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matr[i].data()));
    }
    taskDataSeq->inputs_count = {rows, cols};
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(max_example.data()));
    taskDataSeq->outputs_count.emplace_back(max_example.size());
    gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(global_max, max_example);
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}
