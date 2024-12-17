#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/malyshev_v_conjugate_gradient_method/include/ops_mpi.hpp"

namespace malyshev_conjugate_gradient_method {

std::vector<std::vector<double>> generateRandomSymmetricPositiveDefiniteMatrix(uint32_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dis(0.1, 10.0);

  std::vector<std::vector<double>> matrix(size, std::vector<double>(size));

  for (uint32_t i = 0; i < size; i++) {
    for (uint32_t j = 0; j <= i; j++) {
      matrix[i][j] = dis(gen);
      matrix[j][i] = matrix[i][j];
    }
  }

  // Make the matrix positive definite by adding a multiple of the identity matrix
  for (uint32_t i = 0; i < size; i++) {
    matrix[i][i] += size;
  }

  return matrix;
}

std::vector<double> generateRandomVector(uint32_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dis(0.1, 10.0);
  std::vector<double> vector(size);

  for (auto &el : vector) {
    el = dis(gen);
  }

  return vector;
}

}  // namespace malyshev_conjugate_gradient_method

TEST(malyshev_conjugate_gradient_method, test_pipeline_run) {
  uint32_t size = 1000;

  boost::mpi::communicator world;
  std::vector<std::vector<double>> matrix;
  std::vector<double> b;
  std::vector<double> x;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_conjugate_gradient_method::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    matrix = malyshev_conjugate_gradient_method::generateRandomSymmetricPositiveDefiniteMatrix(size);
    b = malyshev_conjugate_gradient_method::generateRandomVector(size);
    x.resize(size, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataPar->inputs_count.push_back(size);
    taskDataPar->inputs_count.push_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
    taskDataPar->outputs_count.push_back(size);
  }

  auto testMpiTaskParallel = std::make_shared<malyshev_conjugate_gradient_method::TestTaskParallel>(taskDataPar);

  ASSERT_TRUE(testMpiTaskParallel->validation());
  ASSERT_TRUE(testMpiTaskParallel->pre_processing());
  ASSERT_TRUE(testMpiTaskParallel->run());
  ASSERT_TRUE(testMpiTaskParallel->post_processing());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(malyshev_conjugate_gradient_method, test_task_run) {
  uint32_t size = 1000;

  boost::mpi::communicator world;
  std::vector<std::vector<double>> matrix;
  std::vector<double> b;
  std::vector<double> x;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_conjugate_gradient_method::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    matrix = malyshev_conjugate_gradient_method::generateRandomSymmetricPositiveDefiniteMatrix(size);
    b = malyshev_conjugate_gradient_method::generateRandomVector(size);
    x.resize(size, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataPar->inputs_count.push_back(size);
    taskDataPar->inputs_count.push_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
    taskDataPar->outputs_count.push_back(size);
  }

  auto testMpiTaskParallel = std::make_shared<malyshev_conjugate_gradient_method::TestTaskParallel>(taskDataPar);

  ASSERT_TRUE(testMpiTaskParallel->validation());
  ASSERT_TRUE(testMpiTaskParallel->pre_processing());
  ASSERT_TRUE(testMpiTaskParallel->run());
  ASSERT_TRUE(testMpiTaskParallel->post_processing());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) ppc::core::Perf::print_perf_statistic(perfResults);
}