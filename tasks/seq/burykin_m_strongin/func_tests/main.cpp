#include <gtest/gtest.h>

#include <vector>

#include "seq/burykin_m_strongin/include/ops_seq.hpp"

TEST(Burykin_M_Strongin_Func, Strongin_Method_Functional) {
  // Создаем данные
  double x0 = -5.0;
  double x1 = 5.0;
  double epsilon = 0.001;
  std::vector<double> out(1, 0.0);

  // Создаем TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x0));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x1));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  burykin_m_strongin::StronginOptimization testClass(taskDataSeq);

  ASSERT_TRUE(testClass.validation());
  ASSERT_TRUE(testClass.pre_processing());
  ASSERT_TRUE(testClass.run());
  ASSERT_TRUE(testClass.post_processing());

  double expected_minimum = 0;
  EXPECT_NEAR(expected_minimum, out[0], epsilon);
}

TEST(Burykin_M_Strongin_Func, Strongin_Method_Small_Range) {
  double x0 = -1.0;
  double x1 = 1.0;
  double epsilon = 0.0001;
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x0));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x1));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  burykin_m_strongin::StronginOptimization testClass(taskDataSeq);

  ASSERT_TRUE(testClass.validation());
  ASSERT_TRUE(testClass.pre_processing());
  ASSERT_TRUE(testClass.run());
  ASSERT_TRUE(testClass.post_processing());

  double expected_minimum = 0;
  EXPECT_NEAR(expected_minimum, out[0], epsilon);
}

TEST(Burykin_M_Strongin_Func, Strongin_Method_Large_Range) {
  double x0 = -100.0;
  double x1 = 100.0;
  double epsilon = 0.001;
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x0));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x1));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  burykin_m_strongin::StronginOptimization testClass(taskDataSeq);

  ASSERT_TRUE(testClass.validation());
  ASSERT_TRUE(testClass.pre_processing());
  ASSERT_TRUE(testClass.run());
  ASSERT_TRUE(testClass.post_processing());

  double expected_minimum = 0;
  EXPECT_NEAR(expected_minimum, out[0], epsilon);
}