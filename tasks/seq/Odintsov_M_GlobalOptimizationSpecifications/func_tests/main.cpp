
#include <gtest/gtest.h>

#include <vector>

#include "seq/Odintsov_M_GlobalOptimizationSpecifications/include/ops_seq.hpp"

TEST(Odintsov_M_GlobalOptimizationSpecifications_seq, test_min_1) {
  // Create data
  double step = 0.5;
  std::vector<double> area = {-10, 10, -10, 10};
  std::vector<double> func = {2, 3};
  std::vector<double> constraint = {1, 1, 1, 2, 1, 1};
  std::vector<double> out = {0};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(area.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(func.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraint.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step));
  taskDataSeq->inputs_count.emplace_back(1);  // Количество ограничений
  taskDataSeq->inputs_count.emplace_back(0);  // Режим

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  Odintsov_M_GlobalOptimizationSpecifications_seq::GlobalOptimizationSpecificationsSequential testClass(taskDataSeq);
  ASSERT_EQ(testClass.validation(), true);
  testClass.pre_processing();
  testClass.run();
  testClass.post_processing();
  ASSERT_EQ(8, out[0]);
}
TEST(Odintsov_M_GlobalOptimizationSpecifications_seq, test_max_1) {
  // Create data
  double step = 0.5;
  std::vector<double> area = {-10, 10, -10, 10};
  std::vector<double> func = {2, 3};
  std::vector<double> constraint = {1, 1, 1, 2, 1, 1};
  std::vector<double> out = {0};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(area.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(func.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraint.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step));
  taskDataSeq->inputs_count.emplace_back(1);  // Количество ограничений
  taskDataSeq->inputs_count.emplace_back(1);  // Режим

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  Odintsov_M_GlobalOptimizationSpecifications_seq::GlobalOptimizationSpecificationsSequential testClass(taskDataSeq);
  ASSERT_EQ(testClass.validation(), true);
  testClass.pre_processing();
  testClass.run();
  testClass.post_processing();
  ASSERT_EQ(313, out[0]);
}

TEST(Odintsov_M_GlobalOptimizationSpecifications_seq, test_min_2) {
  // Create data
  double step = 0.1;
  std::vector<double> area = {-10, 10, -10, 10};
  std::vector<double> func = {5, 5};
  std::vector<double> constraint = {1, 1, 1, 2, 1, 1, 2, 3, 1, 4, 1, 1};
  std::vector<double> out = {0};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(area.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(func.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraint.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step));
  taskDataSeq->inputs_count.emplace_back(4);  // Количество ограничений
  taskDataSeq->inputs_count.emplace_back(0);  // Режим

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  Odintsov_M_GlobalOptimizationSpecifications_seq::GlobalOptimizationSpecificationsSequential testClass(taskDataSeq);
  ASSERT_EQ(testClass.validation(), true);
  testClass.pre_processing();
  testClass.run();
  testClass.post_processing();
  EXPECT_NEAR(46.08, out[0], 0.000001);
}