#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/malyshev_v_conjugate_gradient_method/include/ops_mpi.hpp"

namespace malyshev_v_conjugate_gradient_method {

std::vector<std::vector<double>> getTestMatrix() { return {{4, 1}, {1, 3}}; }

std::vector<double> getTestVector() { return {1, 2}; }

}  // namespace malyshev_v_conjugate_gradient_method

TEST(malyshev_v_conjugate_gradient_method, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<std::vector<double>> matrix;
  std::vector<double> vector;
  std::vector<double> mpiResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_v_conjugate_gradient_method::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    matrix = malyshev_v_conjugate_gradient_method::getTestMatrix();
    vector = malyshev_v_conjugate_gradient_method::getTestVector();
    mpiResult.resize(vector.size());

    for (auto &row : matrix) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
    taskDataPar->inputs_count.push_back(vector.size());
    taskDataPar->inputs_count.push_back(vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiResult.data()));
    taskDataPar->outputs_count.push_back(mpiResult.size());
  }

  auto testMpiTaskParallel = std::make_shared<malyshev_v_conjugate_gradient_method::TestTaskParallel>(taskDataPar);

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

TEST(malyshev_v_conjugate_gradient_method, test_task_run) {
  boost::mpi::communicator world;
  std::vector<std::vector<double>> matrix;
  std::vector<double> vector;
  std::vector<double> mpiResult;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  malyshev_v_conjugate_gradient_method::TestTaskParallel taskMPI(taskDataPar);

  if (world.rank() == 0) {
    matrix = malyshev_v_conjugate_gradient_method::getTestMatrix();
    vector = malyshev_v_conjugate_gradient_method::getTestVector();
    mpiResult.resize(vector.size());

    for (auto &row : matrix) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
    taskDataPar->inputs_count.push_back(vector.size());
    taskDataPar->inputs_count.push_back(vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpiResult.data()));
    taskDataPar->outputs_count.push_back(mpiResult.size());
  }

  auto testMpiTaskParallel = std::make_shared<malyshev_v_conjugate_gradient_method::TestTaskParallel>(taskDataPar);

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