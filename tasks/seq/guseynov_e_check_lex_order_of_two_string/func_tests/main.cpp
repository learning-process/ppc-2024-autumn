#include <gtest/gtest.h>

#include <seq/guseynov_e_check_lex_order_of_two_string/include/ops_seq.hpp>
#include <vector>

TEST(guseynov_e_check_lex_order_of_two_string_seq, Test_equal_words) {
  // create data
  std::vector<std::vector<char>> in = {{'c', 'a', 't'}, {'c', 'a', 't'}};
  std::vector<int> out(1, 0);

  // create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[0].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[1].data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(in[0].size());
  taskDataSeq->inputs_count.emplace_back(in[1].size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // create Task
  guseynov_e_check_lex_order_of_two_string_seq::TestTaskSequential testTaskSequantial(taskDataSeq);
  ASSERT_EQ(testTaskSequantial.validation(), true);
  testTaskSequantial.pre_processing();
  testTaskSequantial.run();
  testTaskSequantial.post_processing();
  ASSERT_EQ(out[0], 0);
}

TEST(guseynov_e_check_lex_order_of_two_string_seq, Test_second_string_is_greater) {
  // create data
  std::vector<std::vector<char>> in = {{'a', 'p', 'p', 'l', 'e'}, {'b', 'a', 'n', 'a', 'n'}};
  std::vector<int> out(1, 0);

  // create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[0].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[1].data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(in[0].size());
  taskDataSeq->inputs_count.emplace_back(in[1].size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // create Task
  guseynov_e_check_lex_order_of_two_string_seq::TestTaskSequential testTaskSequantial(taskDataSeq);
  ASSERT_EQ(testTaskSequantial.validation(), true);
  testTaskSequantial.pre_processing();
  testTaskSequantial.run();
  testTaskSequantial.post_processing();
  ASSERT_EQ(out[0], 1);
}

TEST(guseynov_e_check_lex_order_of_two_string_seq, Test_first_string_is_greater) {
  // create data
  std::vector<std::vector<char>> in = {{'d', 'o', 'g'}, {'c', 'a', 't'}};
  std::vector<int> out(1, 0);

  // create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[0].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[1].data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(in[0].size());
  taskDataSeq->inputs_count.emplace_back(in[1].size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // create Task
  guseynov_e_check_lex_order_of_two_string_seq::TestTaskSequential testTaskSequantial(taskDataSeq);
  ASSERT_EQ(testTaskSequantial.validation(), true);
  testTaskSequantial.pre_processing();
  testTaskSequantial.run();
  testTaskSequantial.post_processing();
  ASSERT_EQ(out[0], 2);
}

TEST(guseynov_e_check_lex_order_of_two_string_seq, Test_first_string_is_prefix) {
  // create data
  std::vector<std::vector<char>> in = {{'a', 'b', 'c'}, {'a', 'b', 'c', 'd'}};
  std::vector<int> out(1, 0);

  // create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[0].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[1].data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(in[0].size());
  taskDataSeq->inputs_count.emplace_back(in[1].size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // create Task
  guseynov_e_check_lex_order_of_two_string_seq::TestTaskSequential testTaskSequantial(taskDataSeq);
  ASSERT_EQ(testTaskSequantial.validation(), true);
  testTaskSequantial.pre_processing();
  testTaskSequantial.run();
  testTaskSequantial.post_processing();
  ASSERT_EQ(out[0], 1);
}

TEST(guseynov_e_check_lex_order_of_two_string_seq, Test_second_string_is_prefix) {
  // create data
  std::vector<std::vector<char>> in = {{'a', 'b', 'c', 'd'}, {'a', 'b', 'c'}};
  std::vector<int> out(1, 0);

  // create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[0].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[1].data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(in[0].size());
  taskDataSeq->inputs_count.emplace_back(in[1].size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // create Task
  guseynov_e_check_lex_order_of_two_string_seq::TestTaskSequential testTaskSequantial(taskDataSeq);
  ASSERT_EQ(testTaskSequantial.validation(), true);
  testTaskSequantial.pre_processing();
  testTaskSequantial.run();
  testTaskSequantial.post_processing();
  ASSERT_EQ(out[0], 2);
}