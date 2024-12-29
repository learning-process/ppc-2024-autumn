#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/polikanov_v_gauss_band_columns/include/ops_mpi.hpp"

namespace polikanov_v_gauss_band_columns_mpi {

void generateGaussianCompatibleMatrix(int size, std::vector<double>& extendedMatrix) {
  std::vector<double> solutions(size);
  extendedMatrix.resize(size * (size + 1));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(-100.0, 100.0);

  for (int i = 0; i < size; ++i) {
    solutions[i] = dist(gen);
  }

  for (int i = 0; i < size; ++i) {
    double sum = 0.0;
    for (int j = 0; j < size; ++j) {
      double value = dist(gen);
      extendedMatrix[i * (size + 1) + j] = value;
      sum += value * solutions[j];
    }
    extendedMatrix[i * (size + 1) + size] = sum;
  }
}

}  // namespace polikanov_v_gauss_band_columns_mpi

TEST(polikanov_v_gauss_band_columns_mpi, pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  size_t n = 200;
  std::vector<double> input_matrix(n * (n + 1));
  std::vector<double> global_result(n);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    polikanov_v_gauss_band_columns_mpi::generateGaussianCompatibleMatrix(n, input_matrix);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
    taskDataPar->inputs_count.emplace_back(input_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<polikanov_v_gauss_band_columns_mpi::GaussBandColumnsParallelMPI>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    std::vector<double> seq_results(global_result.size());

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(input_matrix.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_results.data()));
    taskDataSeq->outputs_count.emplace_back(seq_results.size());

    auto taskSequential =
        std::make_shared<polikanov_v_gauss_band_columns_mpi::GaussBandColumnsSequentialMPI>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();

    ASSERT_EQ(seq_results.size(), global_result.size());
    for (size_t i = 0; i < seq_results.size(); i++) {
      EXPECT_NEAR(global_result[i], seq_results[i], 0.01);
    }
  }
}

TEST(polikanov_v_gauss_band_columns_mpi, task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  size_t n = 200;
  std::vector<double> input_matrix(n * (n + 1));
  std::vector<double> global_result(n);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    polikanov_v_gauss_band_columns_mpi::generateGaussianCompatibleMatrix(n, input_matrix);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
    taskDataPar->inputs_count.emplace_back(input_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<polikanov_v_gauss_band_columns_mpi::GaussBandColumnsParallelMPI>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    std::vector<double> seq_results(global_result.size());

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(input_matrix.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_results.data()));
    taskDataSeq->outputs_count.emplace_back(seq_results.size());

    auto taskSequential =
        std::make_shared<polikanov_v_gauss_band_columns_mpi::GaussBandColumnsSequentialMPI>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();

    ASSERT_EQ(seq_results.size(), global_result.size());
    for (size_t i = 0; i < seq_results.size(); i++) {
      EXPECT_NEAR(global_result[i], seq_results[i], 0.01);
    }
  }
}
