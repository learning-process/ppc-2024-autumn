// Filateva Elizaveta Number_of_sentences_per_line
#include <gtest/gtest.h>

#include <vector>

#include "seq/filateva_e_number_sentences_line/include/ops_seq.hpp"

TEST(filateva_e_number_sentences_line_seq, one_sentence_line_1) {
  // Create data
  std::string line = "Hello world.";
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(line.data()));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  filateva_e_number_sentences_line_seq::NumberSentencesLine NumS(taskDataSeq);
  ASSERT_EQ(NumS.validation(), true);
  NumS.pre_processing();
  NumS.run();
  NumS.post_processing();
  ASSERT_EQ(1, out[0]);
}

TEST(filateva_e_number_sentences_line_seq, one_sentence_line_2) {
  // Create data
  std::string line = "Hello world";
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(line.data()));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  filateva_e_number_sentences_line_seq::NumberSentencesLine NumS(taskDataSeq);
  ASSERT_EQ(NumS.validation(), true);
  NumS.pre_processing();
  NumS.run();
  NumS.post_processing();
  ASSERT_EQ(1, out[0]);
}

TEST(filateva_e_number_sentences_line_seq, one_sentence_line_3) {
  // Create data
  std::string line = "Hello world!";
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(line.data()));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  filateva_e_number_sentences_line_seq::NumberSentencesLine NumS(taskDataSeq);
  ASSERT_EQ(NumS.validation(), true);
  NumS.pre_processing();
  NumS.run();
  NumS.post_processing();
  ASSERT_EQ(1,out[0]);
}

TEST(filateva_e_number_sentences_line_seq, one_sentence_line_4) {
  // Create data
  std::string line = "Hello world?";
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(line.data()));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  filateva_e_number_sentences_line_seq::NumberSentencesLine NumS(taskDataSeq);
  ASSERT_EQ(NumS.validation(), true);
  NumS.pre_processing();
  NumS.run();
  NumS.post_processing();
  ASSERT_EQ(1, out[0]);
}

TEST(filateva_e_number_sentences_line_seq, several_sentence_line_1) {
  // Create data
  std::string line = "Hello world. How many words are in this sentence? The task of parallel programming.";
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(line.data()));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  filateva_e_number_sentences_line_seq::NumberSentencesLine NumS(taskDataSeq);
  ASSERT_EQ(NumS.validation(), true);
  NumS.pre_processing();
  NumS.run();
  NumS.post_processing();
  ASSERT_EQ(3, out[0]);
}

TEST(filateva_e_number_sentences_line_seq, several_sentence_line_2) {
  // Create data
  std::string line = "Hello world. How many words are in this sentence? The task of parallel programming";
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(line.data()));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  filateva_e_number_sentences_line_seq::NumberSentencesLine NumS(taskDataSeq);
  ASSERT_EQ(NumS.validation(), true);
  NumS.pre_processing();
  NumS.run();
  NumS.post_processing();
  ASSERT_EQ(3, out[0]);
}

TEST(filateva_e_number_sentences_line_seq, empty_string) {
  // Create data
  std::string line = "";
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(line.data()));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  filateva_e_number_sentences_line_seq::NumberSentencesLine NumS(taskDataSeq);
  ASSERT_EQ(NumS.validation(), true);
  NumS.pre_processing();
  NumS.run();
  NumS.post_processing();
  ASSERT_EQ(0, out[0]);
}
