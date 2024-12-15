#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/malyshev_v_conjugate_gradient_method/include/ops_mpi.hpp"

namespace malyshev_v_conjugate_gradient_method_test_function {

std::vector<std::vector<double>> getRandomMatrix(uint32_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<std::vector<double>> matrix(size, std::vector<double>(size));

  for (auto &row : matrix) {
    for (auto &el : row) {
      el = dist(gen);
    }
  }

  return matrix;
}

std::vector<double> getRandomVector(uint32_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> vector(size);

  for (auto &el : vector) {
    el = dist(gen);
  }

  return vector;
}

}  // namespace malyshev_v_conjugate_gradient_method_test_function

TEST(malyshev_v_conjugate_gradient_method_mpi, test_pipeline_run) {
  uint32_t size = 3000;

  boost::mpi::communicator world;
  std::vector<std::vector<double>> randomMatrix;
  std::vector<double> randomVector;
  std::vector<double> mpiSolution;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_v_conjugate_gradient_method_mpi::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomMatrix = malyshev_v_conjugate_gradient_method_test_function::getRandomMatrix(size);
    randomVector = malyshev_v_conjugate_gradient_method_test_function::getRandomVector(size);
    mpiSolution.resize(size);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomMatrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataPar->inputs_count.push_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiSolution.data()));
    taskDataPar->outputs_count.push_back(size);
  }

  auto testMpiTaskParallel = std::make_shared<malyshev_v_conjugate_gradient_method_mpi::TestTaskParallel>(taskDataPar);

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

TEST(malyshev_v_conjugate_gradient_method_mpi, test_task_run) {
  uint32_t size = 3000;

  boost::mpi::communicator world;
  std::vector<std::vector<double>> randomMatrix;
  std::vector<double> randomVector;
  std::vector<double> mpiSolution;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_v_conjugate_gradient_method_mpi::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomMatrix = malyshev_v_conjugate_gradient_method_test_function::getRandomMatrix(size);
    randomVector = malyshev_v_conjugate_gradient_method_test_function::getRandomVector(size);
    mpiSolution.resize(size);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomMatrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataPar->inputs_count.push_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiSolution.data()));
    taskDataPar->outputs_count.push_back(size);
  }

  auto testMpiTaskParallel = std::make_shared<malyshev_v_conjugate_gradient_method_mpi::TestTaskParallel>(taskDataPar);

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