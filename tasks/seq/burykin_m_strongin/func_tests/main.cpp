#include <gtest/gtest.h>

#include "seq/burykin_m_strongin/include/ops_seq.hpp"

TEST(burykin_m_strongin_SEQ, Test_1_1000_001) {
  double x0 = 1.0;
  double x1 = 1000.0;
  double eps = 0.01;
  double res = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x0));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x1));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res));
  taskDataPar->outputs_count.emplace_back(1);

  auto fn = [](double x) { return x * x; };
  burykin_m_strongin::StronginSequential testMpiTaskParallel(taskDataPar, fn);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  // Create data
  double ref_res = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x0));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x1));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&ref_res));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  burykin_m_strongin::StronginSequential testMpiTaskSequential(taskDataSeq, fn);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  ASSERT_NEAR(ref_res, res, 0.2);
}

TEST(burykin_m_strongin_SEQ, Test_1000_1_001) {
  double x0 = 1000.0;
  double x1 = 1.0;
  double eps = 0.01;
  double res = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x0));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x1));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res));
  taskDataPar->outputs_count.emplace_back(1);

  auto fn = [](double x) { return x * x; };
  burykin_m_strongin::StronginSequential testMpiTaskParallel(taskDataPar, fn);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  // Create data
  double ref_res = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x0));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x1));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&ref_res));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  burykin_m_strongin::StronginSequential testMpiTaskSequential(taskDataSeq, fn);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  ASSERT_NEAR(ref_res, res, 0.2);
}

TEST(burykin_m_strongin_SEQ, Test_200_700_001) {
  double x0 = 200.0;
  double x1 = 700.0;
  double eps = 0.01;
  double res = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x0));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x1));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res));
  taskDataPar->outputs_count.emplace_back(1);

  auto fn = [](double x) { return x * x; };
  burykin_m_strongin::StronginSequential testMpiTaskParallel(taskDataPar, fn);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  // Create data
  double ref_res = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x0));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x1));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&ref_res));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  burykin_m_strongin::StronginSequential testMpiTaskSequential(taskDataSeq, fn);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  ASSERT_NEAR(ref_res, res, 0.2);
}

TEST(burykin_m_strongin_SEQ, Test_0_10_0001) {
  double x0 = 0.0;
  double x1 = 10.0;
  double eps = 0.001;
  double res = 0;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x0));
  taskData->inputs_count.emplace_back(1);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x1));
  taskData->inputs_count.emplace_back(1);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
  taskData->inputs_count.emplace_back(1);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res));
  taskData->outputs_count.emplace_back(1);

  auto fn = [](double x) { return x * x; };
  burykin_m_strongin::StronginSequential testTask(taskData, fn);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  double expected_res = 0;
  ASSERT_NEAR(expected_res, res, 0.001);
}

TEST(burykin_m_strongin_SEQ, Test_0_5_00001) {
  double x0 = 0.0;
  double x1 = 10.0;
  double eps = 0.0001;
  double res = 0;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x0));
  taskData->inputs_count.emplace_back(1);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x1));
  taskData->inputs_count.emplace_back(1);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
  taskData->inputs_count.emplace_back(1);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res));
  taskData->outputs_count.emplace_back(1);

  auto fn = [](double x) { return x * 3; };
  burykin_m_strongin::StronginSequential testTask(taskData, fn);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  double expected_res = 0;
  ASSERT_NEAR(expected_res, res, 0.001);
}