#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/savchenko_m_matrix_mult_strassen/include/ops_mpi_savchenko_L3.hpp"

namespace savchenko_m_matrix_mult_strassen_mpi {
std::vector<double> getRandomMatrix(size_t size, double min, double max) {
  if (size <= 0) {
    throw std::out_of_range("size must be greater than 0");
  }
  if (min > max) {
    throw std::invalid_argument("min should not be greater than max");
  }

  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dist(min, max);

  // Forming a random matrix
  std::vector<double> matrix(size * size);
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      matrix[i * size + j] = dist(gen);
    }
  }

  return matrix;
}

double getRandomDouble(double min, double max) {
  if (min > max) {
    throw std::invalid_argument("min should not be greater than max");
  }

  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dist(min, max);

  return dist(gen);
}

}  // namespace savchenko_m_matrix_mult_strassen_mpi

TEST(savchenko_m_matrix_mult_strassen_mpi, test_pipeline_run) {
  // Create data
  boost::mpi::communicator world;

  size_t size = 256;

  double gen_min = -10.0;
  double gen_max = 10.0;

  std::vector<double> matrix_A(size * size, 0.0);
  std::vector<double> matrix_B(size * size, 0.0);
  std::vector<double> matrix_res(size * size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    matrix_A = savchenko_m_matrix_mult_strassen_mpi::getRandomMatrix(size, gen_min, gen_max);
    matrix_B = savchenko_m_matrix_mult_strassen_mpi::getRandomMatrix(size, gen_min, gen_max);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(size);
  }

  auto testMpiTaskParallel = std::make_shared<savchenko_m_matrix_mult_strassen_mpi::TestMPITaskParallel>(taskDataPar);

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
  }
}

TEST(savchenko_m_matrix_mult_strassen_mpi, test_task_run) {
  // Create data
  boost::mpi::communicator world;

  size_t size = 256;

  double gen_min = -10.0;
  double gen_max = 10.0;

  std::vector<double> matrix_A(size * size, 0.0);
  std::vector<double> matrix_B(size * size, 0.0);
  std::vector<double> matrix_res(size * size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    matrix_A = savchenko_m_matrix_mult_strassen_mpi::getRandomMatrix(size, gen_min, gen_max);
    matrix_B = savchenko_m_matrix_mult_strassen_mpi::getRandomMatrix(size, gen_min, gen_max);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(size);
  }

  auto testMpiTaskParallel = std::make_shared<savchenko_m_matrix_mult_strassen_mpi::TestMPITaskParallel>(taskDataPar);

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
  }
}
