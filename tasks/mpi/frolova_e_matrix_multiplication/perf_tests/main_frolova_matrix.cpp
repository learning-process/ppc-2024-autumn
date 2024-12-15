// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/frolova_e_matrix_multiplication/include/ops_mpi_frolova_matrix.hpp"

namespace frolova_e_matrix_multiplication_mpi_test {
std::vector<int> getRandomVector(int size_) {
  if (size_ < 0) {
    return std::vector<int>();
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-100, 100);

  std::vector<int> randomVector(size_);
  std::generate(randomVector.begin(), randomVector.end(), [&]() { return static_cast<int>(dist(gen)); });

  return randomVector;
}
}  // namespace frolova_e_matrix_multiplication_mpi_test

TEST(frolova_e_matrix_multiplication_mpi, test_pipeline_run) {
  // creare data
  boost::mpi::communicator world;
  std::vector<int> values_1 = {600, 600};
  std::vector<int> values_2 = {600, 600};
  std::vector<int> matrixA_;

  std::vector<int> matrixB_;

  std::vector<int32_t> res(360000);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrixA_ = frolova_e_matrix_multiplication_mpi_test::getRandomVector(360000);
    matrixB_ = frolova_e_matrix_multiplication_mpi_test::getRandomVector(360000);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_1.data()));
    taskDataPar->inputs_count.emplace_back(values_1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_2.data()));
    taskDataPar->inputs_count.emplace_back(values_2.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA_.data()));
    taskDataPar->inputs_count.emplace_back(matrixA_.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB_.data()));
    taskDataPar->inputs_count.emplace_back(matrixB_.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<frolova_e_matrix_multiplication_mpi::matrixMultiplicationParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  //// Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  //// Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  //// Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    // Create data

    ppc::core::Perf::print_perf_statistic(perfResults);

    std::vector<int32_t> reference_matrix(360000);

    // Create TaskData

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_1.data()));
    taskDataSeq->inputs_count.emplace_back(values_1.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_2.data()));
    taskDataSeq->inputs_count.emplace_back(values_2.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA_.data()));
    taskDataSeq->inputs_count.emplace_back(matrixA_.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB_.data()));
    taskDataSeq->inputs_count.emplace_back(matrixB_.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_matrix.data()));
    taskDataSeq->outputs_count.emplace_back(reference_matrix.size());

    // Create Task
    frolova_e_matrix_multiplication_mpi::matrixMultiplicationSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_matrix, res);
  }
}

TEST(frolova_e_matrix_multiplication_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> values_1 = {600, 600};
  std::vector<int> values_2 = {600, 600};
  std::vector<int> matrixA_;
  std::vector<int> matrixB_;
  std::vector<int32_t> res(360000);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrixA_ = frolova_e_matrix_multiplication_mpi_test::getRandomVector(360000);
    matrixB_ = frolova_e_matrix_multiplication_mpi_test::getRandomVector(360000);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_1.data()));
    taskDataPar->inputs_count.emplace_back(values_1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_2.data()));
    taskDataPar->inputs_count.emplace_back(values_2.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA_.data()));
    taskDataPar->inputs_count.emplace_back(matrixA_.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB_.data()));
    taskDataPar->inputs_count.emplace_back(matrixB_.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<frolova_e_matrix_multiplication_mpi::matrixMultiplicationParallel>(taskDataPar);
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
    std::vector<int32_t> reference_matrix(360000);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_1.data()));
    taskDataSeq->inputs_count.emplace_back(values_1.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_2.data()));
    taskDataSeq->inputs_count.emplace_back(values_2.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA_.data()));
    taskDataSeq->inputs_count.emplace_back(matrixA_.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB_.data()));
    taskDataSeq->inputs_count.emplace_back(matrixB_.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_matrix.data()));
    taskDataSeq->outputs_count.emplace_back(reference_matrix.size());

    // Create Task
    frolova_e_matrix_multiplication_mpi::matrixMultiplicationSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_matrix, res);
  }
}