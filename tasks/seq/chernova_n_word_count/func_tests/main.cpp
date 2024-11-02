#include <gtest/gtest.h>

#include "seq/chernova_n_word_count/include/ops_seq.hpp"

TEST(Sequential_chernova_n_word_count, Test_five_words) {
  std::vector<char> in;
  std::string testString = "This is a test phrase";
  for (unsigned long int j = 0; j < testString.length(); j++) {
    in.push_back(testString[j]);
  }
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  chernova_n_word_count_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out[0], 5);
}

TEST(Sequential_chernova_n_word_count, Test_five_words_with_space_and_hyphen) {
  std::vector<char> in;
  std::string testString = "This   is a - test phrase";
  for (unsigned long int j = 0; j < testString.length(); j++) {
    in.push_back(testString[j]);
  }
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  chernova_n_word_count_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out[0], 5);
}

TEST(Sequential_chernova_n_word_count, Test_ten_words) {
  std::vector<char> in;
  std::string testString = "This is a test phrase, I really love this phrase";
  for (unsigned long int j = 0; j < testString.length(); j++) {
    in.push_back(testString[j]);
  }
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  chernova_n_word_count_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out[0], 10);
}

TEST(Sequential_chernova_n_word_count, Test_five_words_with_a_lot_of_space) {
  std::vector<char> in;
  std::string testString = "This               is           a             test                phrase";
  for (unsigned long int j = 0; j < testString.length(); j++) {
    in.push_back(testString[j]);
  }
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  chernova_n_word_count_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out[0], 5);
}

TEST(Sequential_chernova_n_word_count, Test_twenty_words) {
  std::vector<char> in;
  std::string testString = "This is a test phrase, I really love this phrase. This is a test phrase, I really love this phrase";
  for (unsigned long int j = 0; j < testString.length(); j++) {
    in.push_back(testString[j]);
  }
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  chernova_n_word_count_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out[0], 20);
}

TEST(Sequential_chernova_n_word_count, Test_five_words_with_space_in_the_end) {
  std::vector<char> in;
  std::string testString = "This is a test phrase           ";
  for (unsigned long int j = 0; j < testString.length(); j++) {
    in.push_back(testString[j]);
  }
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  chernova_n_word_count_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out[0], 5);
}