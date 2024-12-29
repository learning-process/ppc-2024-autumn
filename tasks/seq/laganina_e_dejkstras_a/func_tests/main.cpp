#include <gtest/gtest.h>

#include <random>

#include "seq/laganina_e_dejkstras_a/include/ops_seq.hpp"

TEST(laganina_e_dejkstras_a, Test_validation) {
  int v_ = 0;
  std::vector<int> graph = {0, 1, 2, 0, 0, 1, 0, 0, 0};

  // Create data
  std::vector<int> expectResult = {0, 1, 2};
  std::vector<int> trueResult = {0, 0, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(v_);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(trueResult.data()));
  taskDataSeq->outputs_count.emplace_back(trueResult.size());

  // Create Task
  laganina_e_dejkstras_a_Seq::laganina_e_dejkstras_a_Seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(laganina_e_dejkstras_a, Test_4_1) {
  int v_ = 4;
  std::vector<int> graph = {0, 7, 5, 3, 7, 0, 3, 1, 5, 3, 0, 4, 3, 1, 4, 0};

  // Create data
  std::vector<int> expectResult = {0, 4, 5, 3};
  std::vector<int> trueResult = {0, 0, 0, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(v_);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(trueResult.data()));
  taskDataSeq->outputs_count.emplace_back(trueResult.size());

  // Create Task
  laganina_e_dejkstras_a_Seq::laganina_e_dejkstras_a_Seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expectResult, trueResult);
}
TEST(laganina_e_dejkstras_a, Test_3_1) {
  int v_ = 3;
  std::vector<int> graph = {0, 1, 2, 0, 0, 1, 0, 0, 0};

  // Create data
  std::vector<int> expectResult = {0, 1, 2};
  std::vector<int> trueResult = {0, 0, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(v_);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(trueResult.data()));
  taskDataSeq->outputs_count.emplace_back(trueResult.size());

  // Create Task
  laganina_e_dejkstras_a_Seq::laganina_e_dejkstras_a_Seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expectResult, trueResult);
}
TEST(laganina_e_dejkstras_a, Test_5_1) {
  int v_ = 5;
  std::vector<int> graph = {0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};

  // Create data
  std::vector<int> expectResult = {0, 1, 2, 1, 2};
  std::vector<int> trueResult = {0, 0, 0, 0, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(v_);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(trueResult.data()));
  taskDataSeq->outputs_count.emplace_back(trueResult.size());

  // Create Task
  laganina_e_dejkstras_a_Seq::laganina_e_dejkstras_a_Seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expectResult, trueResult);
}
TEST(laganina_e_dejkstras_a, Test_4_2) {
  int v_ = 4;
  std::vector<int> graph = {0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0};

  // Create data
  std::vector<int> expectResult = {0, 1, 1, 2};
  std::vector<int> trueResult = {0, 0, 0, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(v_);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(trueResult.data()));
  taskDataSeq->outputs_count.emplace_back(trueResult.size());

  // Create Task
  laganina_e_dejkstras_a_Seq::laganina_e_dejkstras_a_Seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expectResult, trueResult);
}
TEST(laganina_e_dejkstras_a, Test_3_2) {
  int v_ = 3;
  std::vector<int> graph = {0, 2, 6, 2, 0, 1, 6, 1, 0};

  // Create data
  std::vector<int> expectResult = {0, 2, 3};
  std::vector<int> trueResult = {0, 0, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(v_);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(trueResult.data()));
  taskDataSeq->outputs_count.emplace_back(trueResult.size());

  // Create Task
  laganina_e_dejkstras_a_Seq::laganina_e_dejkstras_a_Seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expectResult, trueResult);
}
TEST(laganina_e_dejkstras_a, Test_5_2) {
  int v_ = 5;
  std::vector<int> graph = {0, 4, 0, 1, 0, 4, 0, 2, 0, 0, 0, 2, 0, 0, 3, 1, 0, 0, 0, 5, 0, 0, 3, 5, 0};

  // Create data
  std::vector<int> expectResult = {0, 4, 6, 1, 6};
  std::vector<int> trueResult = {0, 0, 0, 0, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(v_);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(trueResult.data()));
  taskDataSeq->outputs_count.emplace_back(trueResult.size());

  // Create Task
  laganina_e_dejkstras_a_Seq::laganina_e_dejkstras_a_Seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expectResult, trueResult);
}
TEST(laganina_e_dejkstras_a, Test_v_less_1) {
  int v_ = 0;
  std::vector<int> graph = {0, 4, 0, 1, 0, 4, 0, 2, 0, 0, 0, 2, 0, 0, 3, 1, 0, 0, 0, 5, 0, 0, 3, 5, 0};

  // Create data
  std::vector<int> expectResult = {0, 4, 6, 1, 6};
  std::vector<int> trueResult = {0, 0, 0, 0, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(v_);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(trueResult.data()));
  taskDataSeq->outputs_count.emplace_back(trueResult.size());

  // Create Task
  laganina_e_dejkstras_a_Seq::laganina_e_dejkstras_a_Seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}
TEST(laganina_e_dejkstras_a, Test_linear) {
  int v_ = 5;
  std::vector<int> graph = {0, 3, 0, 0, 0, 3, 0, 5, 0, 0, 0, 5, 0, 2, 0, 0, 0, 2, 0, 7, 0, 0, 0, 7, 0};

  // Create data
  std::vector<int> expectResult = {0, 3, 8, 10, 17};
  std::vector<int> trueResult = {0, 0, 0, 0, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(v_);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(trueResult.data()));
  taskDataSeq->outputs_count.emplace_back(trueResult.size());

  // Create Task
  laganina_e_dejkstras_a_Seq::laganina_e_dejkstras_a_Seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expectResult, trueResult);
}
