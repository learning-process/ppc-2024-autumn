#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/malyshev_v_monte_carlo_integration/include/ops_seq.hpp"

using namespace std::chrono;

TEST(malyshev_v_monte_carlo_integration, HighSampleCountPerfTest) {
  std::vector<double> global_result(1, 0.0);
  double a = 0.0;
  double b = 1.0;
  double epsilon = 0.00001;

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));

  malyshev_v_monte_carlo_integration::TestMPITaskSequential testTask(taskDataSeq);
  ASSERT_EQ(testTask.validation(), true);

  auto start = high_resolution_clock::now();
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
  auto end = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(end - start).count();

  std::cout << "HighSampleCountPerfTest duration: " << duration << " ms\n";
}

TEST(malyshev_v_monte_carlo_integration, MinimalRangePerfTest) {
  std::vector<double> global_result(1, 0.0);
  double a = 0.0;
  double b = 0.1;
  double epsilon = 0.001;

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));

  malyshev_v_monte_carlo_integration::TestMPITaskSequential testTask(taskDataSeq);
  ASSERT_EQ(testTask.validation(), true);

  auto start = high_resolution_clock::now();
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
  auto end = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(end - start).count();

  std::cout << "MinimalRangePerfTest duration: " << duration << " ms\n";
}

TEST(malyshev_v_monte_carlo_integration, ExtendedRangePerfTest) {
  std::vector<double> global_result(1, 0.0);
  double a = 0.0;
  double b = 10.0;
  double epsilon = 0.001;

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));

  malyshev_v_monte_carlo_integration::TestMPITaskSequential testTask(taskDataSeq);
  ASSERT_EQ(testTask.validation(), true);

  auto start = high_resolution_clock::now();
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
  auto end = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(end - start).count();

  std::cout << "ExtendedRangePerfTest duration: " << duration << " ms\n";
}

TEST(malyshev_v_monte_carlo_integration, SmallEpsilonPerfTest) {
  std::vector<double> global_result(1, 0.0);
  double a = 0.0;
  double b = 1.0;
  double epsilon = 0.00001;

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));

  malyshev_v_monte_carlo_integration::TestMPITaskSequential testTask(taskDataSeq);
  ASSERT_EQ(testTask.validation(), true);

  auto start = high_resolution_clock::now();
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
  auto end = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(end - start).count();

  std::cout << "SmallEpsilonPerfTest duration: " << duration << " ms\n";
}

TEST(malyshev_v_monte_carlo_integration, NegativeRangePerfTest) {
  std::vector<double> global_result(1, 0.0);
  double a = -1.0;
  double b = 1.0;
  double epsilon = 0.0004;

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));

  malyshev_v_monte_carlo_integration::TestMPITaskSequential testTask(taskDataSeq);
  ASSERT_EQ(testTask.validation(), true);

  auto start = high_resolution_clock::now();
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();
  auto end = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(end - start).count();

  std::cout << "NegativeRangePerfTest duration: " << duration << " ms\n";
}
