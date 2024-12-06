#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include <random>
#include "mpi/sedova_o_vertical_ribbon_scheme/include/ops_mpi.hpp"

std::vector<int> generateVector(int count) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vector(count);
  for (int i = 0; i < count; i++) {
    vector[i] = (gen() % 100) - 33;
  }
  return vector;
}

std::vector<int> generateMatrix(int rows, int cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  int count = rows * cols;
  std::vector<int> matrix(count);
  for (int i = 0; i < count; i++) {
    matrix[i] = (gen() % 100) - 33;
  }
  return matrix;
}

std::vector<int> testMatrix = generateMatrix(1240, 1000);
std::vector<int> testVector = generateVector(1240);

TEST(sedova_o_vertical_ribbon_scheme_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix = testMatrix;
  std::vector<int> global_vector = testVector;
  std::vector<int> global_result(1000, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataParallel->inputs_count.emplace_back(1240);
    taskDataParallel->inputs_count.emplace_back(1000);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataParallel->inputs_count.emplace_back(global_vector.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
    taskDataParallel->outputs_count.emplace_back(global_result.size());
  }

  auto testTask = std::make_shared<sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI>(taskDataParallel);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(sedova_o_vertical_ribbon_scheme_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix = testMatrix;
  std::vector<int> global_vector = testVector;
  std::vector<int> global_result(1240, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataParallel->inputs_count.emplace_back(1240);
    taskDataParallel->inputs_count.emplace_back(1000);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataParallel->inputs_count.emplace_back(global_vector.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
    taskDataParallel->outputs_count.emplace_back(global_result.size());
  }

  auto testTask = std::make_shared<sedova_o_vertical_ribbon_scheme_mpi::ParallelMPI>(taskDataParallel);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}