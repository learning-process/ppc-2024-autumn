#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/veliev_e_jacobi_method/include/ops_seq.hpp"

TEST(veliev_e_jacobi_method, test_pipeline_run) {
  const uint32_t matrixSize = 500;

  std::vector<double> referenceResult(matrixSize, 1.0);
  std::vector<double> rhsVector(matrixSize, 510.0);
  std::vector<double> resultVector(matrixSize, 0.0);
  std::vector<double> matrixData(matrixSize * matrixSize, 1.0);

  for (uint32_t idx = 0; idx < matrixSize; ++idx) {
    matrixData[idx * matrixSize + idx] = 510.0;
  }

  auto taskContainer = std::make_shared<ppc::core::TaskData>();
  taskContainer->inputs.push_back(reinterpret_cast<uint8_t *>(matrixData.data()));
  taskContainer->inputs.push_back(reinterpret_cast<uint8_t *>(rhsVector.data()));
  taskContainer->inputs.push_back(reinterpret_cast<uint8_t *>(resultVector.data()));
  taskContainer->inputs_count.push_back(matrixSize);
  taskContainer->inputs_count.push_back(rhsVector.size());
  taskContainer->inputs_count.push_back(resultVector.size());
  taskContainer->outputs.push_back(reinterpret_cast<uint8_t *>(resultVector.data()));
  taskContainer->outputs_count.push_back(resultVector.size());

  auto jacobiMethodProcessor = std::make_shared<veliev_e_jacobi_method::MethodJacobi>(taskContainer);

  auto performanceAttributes = std::make_shared<ppc::core::PerfAttr>();
  performanceAttributes->num_running = 10;

  const auto initialTimePoint = std::chrono::high_resolution_clock::now();
  performanceAttributes->current_timer = [&] {
    auto currentTime = std::chrono::high_resolution_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::nanoseconds>(currentTime - initialTimePoint).count();
    return static_cast<double>(elapsedTime) * 1e-9;
  };

  auto performanceResultsContainer = std::make_shared<ppc::core::PerfResults>();

  auto performanceAnalyzer = std::make_shared<ppc::core::Perf>(jacobiMethodProcessor);
  performanceAnalyzer->pipeline_run(performanceAttributes, performanceResultsContainer);
  ppc::core::Perf::print_perf_statistic(performanceResultsContainer);

  for (uint32_t idx = 0; idx < matrixSize; ++idx) {
    ASSERT_NEAR(resultVector[idx], referenceResult[idx], 0.5);
  }
}

TEST(veliev_e_jacobi_method, test_task_run) {
  const uint32_t matrixDim = 500;

  std::vector<double> expectedResults(matrixDim, 1.0);
  std::vector<double> rhsVector(matrixDim, 510.0);
  std::vector<double> resultVector(matrixDim, 0.0);
  std::vector<double> matrixData(matrixDim * matrixDim, 1.0);

  for (uint32_t idx = 0; idx < matrixDim; ++idx) {
    matrixData[idx * matrixDim + idx] = 510.0;
  }

  auto taskDataContainer = std::make_shared<ppc::core::TaskData>();
  taskDataContainer->inputs.push_back(reinterpret_cast<uint8_t *>(matrixData.data()));
  taskDataContainer->inputs.push_back(reinterpret_cast<uint8_t *>(rhsVector.data()));
  taskDataContainer->inputs.push_back(reinterpret_cast<uint8_t *>(resultVector.data()));
  taskDataContainer->inputs_count.push_back(matrixDim);
  taskDataContainer->inputs_count.push_back(rhsVector.size());
  taskDataContainer->inputs_count.push_back(resultVector.size());
  taskDataContainer->outputs.push_back(reinterpret_cast<uint8_t *>(resultVector.data()));
  taskDataContainer->outputs_count.push_back(resultVector.size());

  auto jacobiProcessor = std::make_shared<veliev_e_jacobi_method::MethodJacobi>(taskDataContainer);

  auto performanceParams = std::make_shared<ppc::core::PerfAttr>();
  performanceParams->num_running = 10;

  const auto startTime = std::chrono::high_resolution_clock::now();
  performanceParams->current_timer = [&] {
    auto currentTime = std::chrono::high_resolution_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::nanoseconds>(currentTime - startTime).count();
    return static_cast<double>(elapsedTime) * 1e-9;
  };

  auto performanceResultsContainer = std::make_shared<ppc::core::PerfResults>();

  auto performanceEvaluator = std::make_shared<ppc::core::Perf>(jacobiProcessor);
  performanceEvaluator->task_run(performanceParams, performanceResultsContainer);
  ppc::core::Perf::print_perf_statistic(performanceResultsContainer);

  for (uint32_t idx = 0; idx < matrixDim; ++idx) {
    ASSERT_NEAR(resultVector[idx], expectedResults[idx], 0.5);
  }
}
