// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/kozlova_e_lexic_order/include/ops_seq.hpp"

TEST(kozlova_e_lexic_order, Test_twoStrings) {
  // Create data
  const char *str1 = "aaabbbccc";
  const char *str2 = "apples";
  std::vector<int> out(2, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(str1)));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(str2)));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(2));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(static_cast<uint32_t>(2));

  // Create Task
  kozlova_e_lexic_order::StringComparator testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out[0], 1);
  ASSERT_EQ(out[1], 0);
}

TEST(kozlova_e_lexic_order, Test_EQ_strings) {
  // Create data
  const char *str1 = "aaabbbccc";
  const char *str2 = "aaabbbccc";
  std::vector<int> out(2, 0);
  std::vector<int> expect(2, 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(str1)));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(str2)));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(2));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(static_cast<uint32_t>(2));

  // Create Task
  kozlova_e_lexic_order::StringComparator testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out, expect);
}

TEST(kozlova_e_lexic_order, Test_not_eq_strings) {
  // Create data
  const char *str1 = "asd";
  const char *str2 = "qwerty";
  std::vector<int> out(2, 0);
  std::vector<int> expect(2, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(str1)));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(str2)));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(2));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(static_cast<uint32_t>(2));

  // Create Task
  kozlova_e_lexic_order::StringComparator testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out, expect);
}

TEST(kozlova_e_lexic_order, Test_empty_strings) {
  // Create data
  const char *str1 = " ";
  const char *str2 = " ";
  std::vector<int> out(2, 0);
  std::vector<int> expect(2, 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(str1)));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(str2)));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(2));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(static_cast<uint32_t>(2));

  // Create Task
  kozlova_e_lexic_order::StringComparator testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out, expect);
}

TEST(kozlova_e_lexic_order, Test_register_strings) {
  // Create data
  const char *str1 = "aBc";
  const char *str2 = "abC";
  std::vector<int> out(2, 0);
  std::vector<int> expect(2, 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(str1)));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(str2)));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(2));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(static_cast<uint32_t>(2));

  // Create Task
  kozlova_e_lexic_order::StringComparator testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out, expect);
}