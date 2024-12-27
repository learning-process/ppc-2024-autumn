#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/savchenko_m_ribbon_mult_split_a/include/ops_mpi_savchenko.hpp"

namespace savchenko_m_ribbon_mult_split_a_mpi {
std::vector<int> getRandomMatrix(size_t rows, size_t columns, int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());

  // Forming a random matrix
  std::vector<int> matrix(rows * columns);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < columns; j++) {
      matrix[i * columns + j] = min + gen() % (max - min + 1);
    }
  }

  return matrix;
}

int getRandomInt(int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  int rand_int = min + gen() % (max - min + 1);
  return rand_int;
}
}  // namespace savchenko_m_ribbon_mult_split_a_mpi

TEST(savchenko_m_ribbon_mult_split_a_mpi, test_pipeline_run) {
  // Create data
  boost::mpi::communicator world;

  int size;
  int res_size;
  int gen_min;
  int gen_max;

  std::vector<int> matrix_A;
  std::vector<int> matrix_B;
  std::vector<int> matrix_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    size = 516;
    res_size = size * size;

    gen_min = -1000;
    gen_max = 1000;

    matrix_A = savchenko_m_ribbon_mult_split_a_mpi::getRandomMatrix(size, size, gen_min, gen_max);
    matrix_B = savchenko_m_ribbon_mult_split_a_mpi::getRandomMatrix(size, size, gen_min, gen_max);
    matrix_res = std::vector<int>(res_size, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(matrix_res.size());
  }

  auto testMpiTaskParallel = std::make_shared<savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskParallel>(taskDataPar);
  // testMpiTaskParallel->validation();
  // testMpiTaskParallel->pre_processing();
  // testMpiTaskParallel->run();
  // testMpiTaskParallel->post_processing();

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

  // Create refference and comparison
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(savchenko_m_ribbon_mult_split_a_mpi, test_task_run) {
  // Create data
  boost::mpi::communicator world;

  int size;
  int res_size;
  int gen_min;
  int gen_max;

  std::vector<int> matrix_A;
  std::vector<int> matrix_B;
  std::vector<int> matrix_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    size = 516;
    res_size = size * size;

    gen_min = -1000;
    gen_max = 1000;

    matrix_A = savchenko_m_ribbon_mult_split_a_mpi::getRandomMatrix(size, size, gen_min, gen_max);
    matrix_B = savchenko_m_ribbon_mult_split_a_mpi::getRandomMatrix(size, size, gen_min, gen_max);
    matrix_res = std::vector<int>(res_size, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(size);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_res.data()));
    taskDataPar->outputs_count.emplace_back(matrix_res.size());
  }

  auto testMpiTaskParallel = std::make_shared<savchenko_m_ribbon_mult_split_a_mpi::TestMPITaskParallel>(taskDataPar);
  // testMpiTaskParallel->validation();
  // testMpiTaskParallel->pre_processing();
  // testMpiTaskParallel->run();
  // testMpiTaskParallel->post_processing();

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

  //// Create refference and comparison
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}
