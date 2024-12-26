#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/malyshev_v_conjugate_gradient_method/include/ops_mpi.hpp"

namespace malyshev_conjugate_gradient {

std::vector<double> generateRandomVector(uint32_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<double> data(size);

  for (auto &el : data) {
    el = -1000.0 + static_cast<double>(gen()) / static_cast<double>(gen.max()) * (2000.0);
  }

  return data;
}

std::vector<std::vector<double>> generateRandomMatrix(uint32_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<std::vector<double>> data(size, std::vector<double>(size));

  for (uint32_t i = 0; i < size; ++i) {
    for (uint32_t j = 0; j < size; ++j) {
      data[i][j] = -1000.0 + static_cast<double>(gen()) / static_cast<double>(gen.max()) * (2000.0);
      if (i == j) {
        data[i][j] += size * 1000.0;  // Ensure diagonal dominance for positive definiteness
      }
    }
  }

  return data;
}

}  // namespace malyshev_conjugate_gradient

TEST(malyshev_conjugate_gradient, test_pipeline_run) {
  uint32_t size = 1000;

  boost::mpi::communicator world;
  std::vector<std::vector<double>> randomMatrix;
  std::vector<double> randomVector;
  std::vector<double> mpiResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_conjugate_gradient::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomMatrix = malyshev_conjugate_gradient::generateRandomMatrix(size);
    randomVector = malyshev_conjugate_gradient::generateRandomVector(size);
    mpiResult.resize(size);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomMatrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataPar->inputs_count.push_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiResult.data()));
    taskDataPar->outputs_count.push_back(size);
  }

  auto testMpiTaskParallel = std::make_shared<malyshev_conjugate_gradient::TestTaskParallel>(taskDataPar);

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

TEST(malyshev_conjugate_gradient, test_task_run) {
  uint32_t size = 1000;

  boost::mpi::communicator world;
  std::vector<std::vector<double>> randomMatrix;
  std::vector<double> randomVector;
  std::vector<double> mpiResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_conjugate_gradient::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    randomMatrix = malyshev_conjugate_gradient::generateRandomMatrix(size);
    randomVector = malyshev_conjugate_gradient::generateRandomVector(size);
    mpiResult.resize(size);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomMatrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
    taskDataPar->inputs_count.push_back(size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiResult.data()));
    taskDataPar->outputs_count.push_back(size);
  }

  auto testMpiTaskParallel = std::make_shared<malyshev_conjugate_gradient::TestTaskParallel>(taskDataPar);

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