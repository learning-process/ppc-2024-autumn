#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/lopatin_i_strip_horizontal_scheme/include/stripHorizontalSchemeHeaderMPI.hpp"

std::vector<int> generateVector(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> outputVector(size);
  for (int i = 0; i < size; i++) {
    outputVector[i] = (gen() % 200) - 99;
  }
  return outputVector;
}

std::vector<int> generateMatrix(int sizeX, int sizeY) {
  std::random_device dev;
  std::mt19937 gen(dev());
  int matrixSize = sizeX * sizeY;
  std::vector<int> outputMatrix(matrixSize);
  for (int i = 0; i < matrixSize; i++) {
    outputMatrix[i] = (gen() % 200) - 99;
  }
  return outputMatrix;
}

std::vector<int> testMatrix = generateMatrix(2560, 1440);
std::vector<int> testVector = generateVector(2560);

TEST(lopatin_i_strip_horizontal_scheme_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> inputMatrix = testMatrix;
  std::vector<int> inputVector = testVector;
  std::vector<int> resultVector(1440, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrix.data()));
    taskDataParallel->inputs_count.emplace_back(2560);
    taskDataParallel->inputs_count.emplace_back(1440);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
    taskDataParallel->inputs_count.emplace_back(inputVector.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultVector.data()));
    taskDataParallel->outputs_count.emplace_back(resultVector.size());
  }

  auto testTask = std::make_shared<lopatin_i_strip_horizontal_scheme_mpi::TestMPITaskParallel>(taskDataParallel);

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

TEST(lopatin_i_strip_horizontal_scheme_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> inputMatrix = testMatrix;
  std::vector<int> inputVector = testVector;
  std::vector<int> resultVector(2160, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrix.data()));
    taskDataParallel->inputs_count.emplace_back(2560);
    taskDataParallel->inputs_count.emplace_back(1440);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
    taskDataParallel->inputs_count.emplace_back(inputVector.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultVector.data()));
    taskDataParallel->outputs_count.emplace_back(resultVector.size());
  }

  auto testTask = std::make_shared<lopatin_i_strip_horizontal_scheme_mpi::TestMPITaskParallel>(taskDataParallel);

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