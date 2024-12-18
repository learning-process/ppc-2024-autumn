#include <gtest/gtest.h>

#include "seq/kharin_m_number_of_sentences_seq/include/ops_seq.hpp"

TEST(Sequential_Sentences_Count, Test_Simple_Sentences) {
  std::string input_text = "This is sentence one. This is sentence two! Is this sentence three? This is sentence four.";
  std::vector<int> sentence_count(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input_text.c_str())));
  taskDataSeq->inputs_count.emplace_back(input_text.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(sentence_count.data()));
  taskDataSeq->outputs_count.emplace_back(sentence_count.size());
  // Run sequential version
  kharin_m_number_of_sentences_seq::CountSentencesSequential countSentencesSequential(taskDataSeq);
  ASSERT_EQ(countSentencesSequential.validation(), true);
  countSentencesSequential.pre_processing();
  countSentencesSequential.run();
  countSentencesSequential.post_processing();
  // Compare results
  ASSERT_EQ(sentence_count[0], 4);
}

TEST(Sequential_Sentences_Count, Test_Empty_Text) {
  std::string input_text;
  std::vector<int> sentence_count(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input_text.c_str())));
  taskDataSeq->inputs_count.emplace_back(input_text.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(sentence_count.data()));
  taskDataSeq->outputs_count.emplace_back(sentence_count.size());
  // Run sequential version
  kharin_m_number_of_sentences_seq::CountSentencesSequential countSentencesSequential(taskDataSeq);
  ASSERT_EQ(countSentencesSequential.validation(), true);
  countSentencesSequential.pre_processing();
  countSentencesSequential.run();
  countSentencesSequential.post_processing();
  // Compare results
  ASSERT_EQ(sentence_count[0], 0);
}

TEST(Sequential_Sentences_Count, Test_Long_Text) {
  std::string input_text;
  std::vector<int> sentence_count(1, 0);

  for (int i = 0; i < 100; i++) {
    input_text += "This is sentence number " + std::to_string(i + 1) + ". ";
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input_text.c_str())));
  taskDataSeq->inputs_count.emplace_back(input_text.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(sentence_count.data()));
  taskDataSeq->outputs_count.emplace_back(sentence_count.size());
  // Run sequential version
  kharin_m_number_of_sentences_seq::CountSentencesSequential countSentencesSequential(taskDataSeq);
  ASSERT_EQ(countSentencesSequential.validation(), true);
  countSentencesSequential.pre_processing();
  countSentencesSequential.run();
  countSentencesSequential.post_processing();
  // Compare results
  ASSERT_EQ(sentence_count[0], 100);
}

TEST(Sequential_Sentences_Count, Test_Sentences_with_other_symbols) {
  std::string input_text =
      "Hi! What's you're name? My name is Matthew. How are you? I'm fine, thank you. And you? I'm also fine.";
  std::vector<int> sentence_count(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input_text.c_str())));
  taskDataSeq->inputs_count.emplace_back(input_text.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(sentence_count.data()));
  taskDataSeq->outputs_count.emplace_back(sentence_count.size());
  // Run sequential version
  kharin_m_number_of_sentences_seq::CountSentencesSequential countSentencesSequential(taskDataSeq);
  ASSERT_EQ(countSentencesSequential.validation(), true);
  countSentencesSequential.pre_processing();
  countSentencesSequential.run();
  countSentencesSequential.post_processing();
  // Compare results
  ASSERT_EQ(sentence_count[0], 7);
}