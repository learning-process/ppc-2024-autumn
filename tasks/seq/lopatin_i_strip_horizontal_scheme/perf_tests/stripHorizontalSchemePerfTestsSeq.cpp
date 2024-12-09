#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/lopatin_i_strip_horizontal_scheme/include/stripHorizontalSchemeHeaderSeq.hpp"

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

TEST(lopatin_i_strip_horizontal_scheme_seq, test_pipeline_run) {
  std::vector<int> inputMatrix = testMatrix;
  std::vector<int> inputVector = testVector;
  std::vector<int> resultVector(1440, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrix.data()));
  taskDataSequential->inputs_count.emplace_back(2560);
  taskDataSequential->inputs_count.emplace_back(1440);
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
  taskDataSequential->inputs_count.emplace_back(inputVector.size());
  taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultVector.data()));
  taskDataSequential->outputs_count.emplace_back(resultVector.size());

  auto testTask = std::make_shared<lopatin_i_strip_horizontal_scheme_seq::TestTaskSequential>(taskDataSequential);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(lopatin_i_strip_horizontal_scheme_seq, test_task_run) {
  std::vector<int> inputMatrix = testMatrix;
  std::vector<int> inputVector = testVector;
  std::vector<int> resultVector(1440, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrix.data()));
  taskDataSequential->inputs_count.emplace_back(2560);
  taskDataSequential->inputs_count.emplace_back(1440);
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
  taskDataSequential->inputs_count.emplace_back(inputVector.size());
  taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultVector.data()));
  taskDataSequential->outputs_count.emplace_back(resultVector.size());

  auto testTask = std::make_shared<lopatin_i_strip_horizontal_scheme_seq::TestTaskSequential>(taskDataSequential);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}