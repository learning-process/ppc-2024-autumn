#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/belov_a_max_value_of_matrix_elements/include/ops_mpi.hpp"

template <typename T>
std::vector<T> generate_random_matrix(int rows, int cols) {
  std::vector<T> res(rows * cols);

  std::random_device dev;
  std::mt19937 gen(dev());
  for (size_t i = 0; i < res.size(); i++) {
    res[i] = static_cast<T>(gen() % 1000);
  }
  return res;
}

TEST(belov_a_max_value_matrix_mpi_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> dimensions;
  std::vector<double> global_matrix;
  std::vector<double> parallel_max(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    dimensions = std::vector<int>{3, 4};
    global_matrix = generate_random_matrix<double>(dimensions[0], dimensions[1]);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimensions.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallel_max.data()));
    taskDataPar->outputs_count.emplace_back(parallel_max.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<belov_a_max_value_of_matrix_elements_mpi::MaxValueOfMatrixElementsParallel<double>>(taskDataPar);

  ASSERT_TRUE(testMpiTaskParallel->validation());
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
    std::vector<double> sequence_max(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimensions.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(sequence_max.data()));
    taskDataSeq->outputs_count.emplace_back(sequence_max.size());
    auto testMpiTaskSequential =
        belov_a_max_value_of_matrix_elements_mpi::MaxValueOfMatrixElementsSequential<double>(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ppc::core::Perf::print_perf_statistic(perfResults);
    EXPECT_NEAR(sequence_max[0], parallel_max[0], 1e-5);
  }
}

TEST(belov_a_max_value_matrix_mpi_perf_test, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> dimensions;
  std::vector<double> global_matrix;
  std::vector<double> parallel_max(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    dimensions = std::vector<int>{3, 4};
    global_matrix = generate_random_matrix<double>(dimensions[0], dimensions[1]);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimensions.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallel_max.data()));
    taskDataPar->outputs_count.emplace_back(parallel_max.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<belov_a_max_value_of_matrix_elements_mpi::MaxValueOfMatrixElementsParallel<double>>(taskDataPar);

  ASSERT_TRUE(testMpiTaskParallel->validation());
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
    std::vector<double> sequence_max(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(dimensions.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(sequence_max.data()));
    taskDataSeq->outputs_count.emplace_back(sequence_max.size());
    auto testMpiTaskSequential =
        belov_a_max_value_of_matrix_elements_mpi::MaxValueOfMatrixElementsSequential<double>(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ppc::core::Perf::print_perf_statistic(perfResults);
    EXPECT_NEAR(sequence_max[0], parallel_max[0], 1e-5);
  }
}
