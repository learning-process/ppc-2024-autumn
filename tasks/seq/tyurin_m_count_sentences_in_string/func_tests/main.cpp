#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "seq/tyurin_m_count_sentences_in_string/include/ops_seq.hpp"

TEST(tyurin_m_count_sentences_in_string_seq, test_sentence_count_single_sentence) {
  std::string input_str = "This is a single sentence.";
  int expected_count = 1;

  std::vector<std::string> in_str(1, input_str);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<unsigned char*>(in_str.data()));
  taskDataSeq->inputs_count.emplace_back(in_str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<unsigned char*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  tyurin_m_count_sentences_in_string_seq::SentenceCountTaskSequential sentenceCountTask(taskDataSeq);
  ASSERT_EQ(sentenceCountTask.validation(), true);
  sentenceCountTask.pre_processing();
  sentenceCountTask.run();
  sentenceCountTask.post_processing();
  ASSERT_EQ(expected_count, out[0]);
}

TEST(tyurin_m_count_sentences_in_string_seq, test_sentence_count_multiple_sentences) {
  std::string input_str = "This is the first sentence. Here is another one! And yet another?";
  int expected_count = 3;

  std::vector<std::string> in_str(1, input_str);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<unsigned char*>(in_str.data()));
  taskDataSeq->inputs_count.emplace_back(in_str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<unsigned char*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  tyurin_m_count_sentences_in_string_seq::SentenceCountTaskSequential sentenceCountTask(taskDataSeq);
  ASSERT_EQ(sentenceCountTask.validation(), true);
  sentenceCountTask.pre_processing();
  sentenceCountTask.run();
  sentenceCountTask.post_processing();
  ASSERT_EQ(expected_count, out[0]);
}

TEST(tyurin_m_count_sentences_in_string_seq, test_sentence_count_no_sentences) {
  std::string input_str = "No sentence endings here";
  int expected_count = 0;

  std::vector<std::string> in_str(1, input_str);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<unsigned char*>(in_str.data()));
  taskDataSeq->inputs_count.emplace_back(in_str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<unsigned char*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  tyurin_m_count_sentences_in_string_seq::SentenceCountTaskSequential sentenceCountTask(taskDataSeq);
  ASSERT_EQ(sentenceCountTask.validation(), true);
  sentenceCountTask.pre_processing();
  sentenceCountTask.run();
  sentenceCountTask.post_processing();
  ASSERT_EQ(expected_count, out[0]);
}

TEST(tyurin_m_count_sentences_in_string_seq, test_sentence_count_empty_string) {
  std::string input_str;
  int expected_count = 0;

  std::vector<std::string> in_str(1, input_str);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<unsigned char*>(in_str.data()));
  taskDataSeq->inputs_count.emplace_back(in_str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<unsigned char*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  tyurin_m_count_sentences_in_string_seq::SentenceCountTaskSequential sentenceCountTask(taskDataSeq);
  ASSERT_EQ(sentenceCountTask.validation(), true);
  sentenceCountTask.pre_processing();
  sentenceCountTask.run();
  sentenceCountTask.post_processing();
  ASSERT_EQ(expected_count, out[0]);
}

TEST(tyurin_m_count_sentences_in_string_seq, test_multiple_consecutive_sentence_endings) {
  std::string input_str = "This is a sentence... And another one?!";
  int expected_count = 2;

  std::vector<std::string> in_str(1, input_str);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<unsigned char*>(in_str.data()));
  taskDataSeq->inputs_count.emplace_back(in_str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<unsigned char*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  tyurin_m_count_sentences_in_string_seq::SentenceCountTaskSequential sentenceCountTask(taskDataSeq);
  ASSERT_EQ(sentenceCountTask.validation(), true);
  sentenceCountTask.pre_processing();
  sentenceCountTask.run();
  sentenceCountTask.post_processing();
  ASSERT_EQ(expected_count, out[0]);
}

TEST(tyurin_m_count_sentences_in_string_seq, test_sentence_count_with_various_whitespaces) {
  std::string input_str = "First sentence.\nSecond sentence!\tThird sentence?";
  int expected_count = 3;

  std::vector<std::string> in_str(1, input_str);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<unsigned char*>(in_str.data()));
  taskDataSeq->inputs_count.emplace_back(in_str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<unsigned char*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  tyurin_m_count_sentences_in_string_seq::SentenceCountTaskSequential sentenceCountTask(taskDataSeq);
  ASSERT_EQ(sentenceCountTask.validation(), true);
  sentenceCountTask.pre_processing();
  sentenceCountTask.run();
  sentenceCountTask.post_processing();
  ASSERT_EQ(expected_count, out[0]);
}

TEST(tyurin_m_count_sentences_in_string_seq, test_only_sentence_endings) {
  std::string input_str = "...?!";
  int expected_count = 0;

  std::vector<std::string> in_str(1, input_str);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<unsigned char*>(in_str.data()));
  taskDataSeq->inputs_count.emplace_back(in_str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<unsigned char*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  tyurin_m_count_sentences_in_string_seq::SentenceCountTaskSequential sentenceCountTask(taskDataSeq);
  ASSERT_EQ(sentenceCountTask.validation(), true);
  sentenceCountTask.pre_processing();
  sentenceCountTask.run();
  sentenceCountTask.post_processing();
  ASSERT_EQ(expected_count, out[0]);
}
