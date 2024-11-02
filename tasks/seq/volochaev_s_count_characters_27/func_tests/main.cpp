#include <gtest/gtest.h>

#include <vector>

#include "seq/volochaev_s_count_characters_27/include/ops_seq.hpp"

TEST(volochaev_s_count_characters_27_seq, Test_0) {
  // Create data
  std::vector<std::string> in = {volochaev_s_count_characters_27_seq::get_random_string(20)};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_count_characters_27_seq::Lab1_27 testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(volochaev_s_count_characters_27_seq, Test_1) {
  // Create data
  std::string s = volochaev_s_count_characters_27_seq::get_random_string(20);
  std::vector<std::string> in(2, s);
  std::vector<int> out(1, 0);

  int ans = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_count_characters_27_seq::Lab1_27 testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out[0]);
}

TEST(volochaev_s_count_characters_27_seq, Test_2) {
  // Create data
  std::string s = volochaev_s_count_characters_27_seq::get_random_string(20);
  std::string s1 = s;

  s1.back() = static_cast<char>((static_cast<int>(s1.back()) + 1) % 256);

  std::vector<std::string> in = {s, s1};
  std::vector<int> out(1, 0);
  int ans = 2;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_count_characters_27_seq::Lab1_27 testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out[0]);
}

TEST(volochaev_s_count_characters_27_seq, Test_3) {
  // Create data

  std::string s = volochaev_s_count_characters_27_seq::get_random_string(6);
  std::string s1 = s.substr(0, 2);

  std::vector<std::string> in = {s, s1};
  std::vector<int> out(1, 0);
  int ans = 6;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_count_characters_27_seq::Lab1_27 testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out[0]);
}

TEST(volochaev_s_count_characters_27_seq, Test_4) {
  // Create data
  std::string s = volochaev_s_count_characters_27_seq::get_random_string(6);
  std::string s1 = s.substr(0, 2);

  std::vector<std::string> in = {s1, s};
  std::vector<int> out(1, 0);
  int ans = 6;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_count_characters_27_seq::Lab1_27 testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out[0]);
}

TEST(volochaev_s_count_characters_27_seq, Test_5) {
  // Create data
  std::string s = volochaev_s_count_characters_27_seq::get_random_string(6);
  std::vector<std::string> in(2, s);
  std::vector<int> out(1, 0);
  int ans = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_count_characters_27_seq::Lab1_27 testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out[0]);
}

TEST(volochaev_s_count_characters_27_seq, Test_6) {
  // Create data
  std::string s = volochaev_s_count_characters_27_seq::get_random_string(7);
  std::vector<std::string> in(2, s);
  std::vector<int> out(1, 0);
  int ans = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_count_characters_27_seq::Lab1_27 testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out[0]);
}