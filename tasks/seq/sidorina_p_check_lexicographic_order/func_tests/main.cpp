// Copyright 2024 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/sidorina_p_check_lexicographic_order/include/ops_seq.hpp"

TEST(sidorina_p_check_lexicographic_order_seq, Test_3_elements) {
  std::vector<std::vector<char>> in = {{'e', 'f', 'g'}, {'e', 'k', 'g'}};
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[0].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[1].data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(in[0].size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  sidorina_p_check_lexicographic_order_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(0, out[0]);
}
TEST(sidorina_p_check_lexicographic_order_seq, Test_difference_1st_element_0) {
  std::vector<std::vector<char>> in = {{'a', 'b', 'c'}, {'d', 'b', 'c'}};
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[0].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[1].data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(in[0].size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  sidorina_p_check_lexicographic_order_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(0, out[0]);
}

TEST(sidorina_p_check_lexicographic_order_seq, Test_difference_1st_element_1) {
  std::vector<std::vector<char>> in = {{'b', 'c', 'g'}, {'a', 'c', 'g'}};
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[0].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[1].data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(in[0].size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  sidorina_p_check_lexicographic_order_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(1, out[0]);
}

TEST(sidorina_p_check_lexicographic_order_seq, Test_difference_2nd_element_1) {
  std::vector<std::vector<char>> in = {{'e', 'c', 'g'}, {'e', 'a', 'g'}};
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[0].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[1].data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(in[0].size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  sidorina_p_check_lexicographic_order_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(1, out[0]);
}
TEST(sidorina_p_check_lexicographic_order_seq, Test_difference_3d_element_1) {
  std::vector<std::vector<char>> in = {{'a', 'b', 'g'}, {'a', 'b', 'a'}};
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[0].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[1].data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(in[0].size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  sidorina_p_check_lexicographic_order_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(1, out[0]);
}