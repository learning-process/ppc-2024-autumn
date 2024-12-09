
#include <gtest/gtest.h>

#include <seq/Odintsov_M_VerticalRibbon_seq/include/ops_seq.hpp>
#include <vector>

TEST(Odintsov_M_VerticalRibbon_seq, sz_10000) {
  std::vector<double> matrixA(10000, 1);
  std::vector<double> vectorB(100, 1);

  std::vector<double> vectorC(100, 100);
  std::vector<double> out(vectorC.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vectorB.data()));
  // [0] - szA [1] - rowA [2] - szB
  taskDataSeq->inputs_count.emplace_back(10000);
  taskDataSeq->inputs_count.emplace_back(100);
  taskDataSeq->inputs_count.emplace_back(100);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  Odintsov_M_VerticalRibbon_seq::VerticalRibbonSequential testClass(taskDataSeq);

  ASSERT_TRUE(testClass.validation());

  testClass.pre_processing();
  testClass.run();
  testClass.post_processing();
  EXPECT_EQ(vectorC, out);
}

TEST(Odintsov_M_VerticalRibbon_seq, sz_40000) {
  std::vector<double> matrixA(40000, 1);
  std::vector<double> vectorB(200, 1);

  std::vector<double> vectorC(200, 200);
  std::vector<double> out(vectorC.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vectorB.data()));
  // [0] - szA [1] - rowA [2] - szB
  taskDataSeq->inputs_count.emplace_back(40000);
  taskDataSeq->inputs_count.emplace_back(200);
  taskDataSeq->inputs_count.emplace_back(200);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  Odintsov_M_VerticalRibbon_seq::VerticalRibbonSequential testClass(taskDataSeq);

  ASSERT_TRUE(testClass.validation());

  testClass.pre_processing();
  testClass.run();
  testClass.post_processing();
  EXPECT_EQ(vectorC, out);
}

TEST(Odintsov_M_VerticalRibbon_seq, df_sz_1200) {
  std::vector<double> matrixA(1200, 1);
  std::vector<double> vectorB(30, 1);
  std::vector<double> vectorC(40, 30);
  std::vector<double> out(vectorC.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vectorB.data()));
  // [0] - szA [1] - rowA [2] - szB
  taskDataSeq->inputs_count.emplace_back(1200);
  taskDataSeq->inputs_count.emplace_back(40);
  taskDataSeq->inputs_count.emplace_back(30);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  Odintsov_M_VerticalRibbon_seq::VerticalRibbonSequential testClass(taskDataSeq);

  ASSERT_TRUE(testClass.validation());

  testClass.pre_processing();
  testClass.run();
  testClass.post_processing();
  EXPECT_EQ(vectorC, out);
}
