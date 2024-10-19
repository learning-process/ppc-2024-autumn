// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/kolodkin_g_sentence_count/include/ops_seq.hpp"

TEST(Sequential, Test_two_sentences) {
  // Create data
  std::string in = "Hello! My name is Grisha!";
  std::vector<char> global_str;
  for (int i = 0; i < in.length(); i++) {
    global_str.push_back(in[i]);
  }
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  kolodkin_g_sentence_count_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out[0], 2);
}
TEST(Sequential, Test_sentences_with_special_symbols) {
  // Create data
  std::string in = "Hello!My name is Grisha! I have two pets: cat,dog,parrot.";
  std::vector<char> global_str;
  for (int i = 0; i < in.length(); i++) {
    global_str.push_back(in[i]);
  }
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  kolodkin_g_sentence_count_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out[0], 3);
}
TEST(Sequential, Test_sentences_with_special_symbols_in_end_of_sentence) {
  // Create data
  std::string in =
      "Hello!My name is Grisha! I have two pets: cat,dog,parrot. What is your name?! How are you!? Well...";
  std::vector<char> global_str;
  for (int i = 0; i < in.length(); i++) {
    global_str.push_back(in[i]);
  }
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  kolodkin_g_sentence_count_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out[0], 6);
}
TEST(Sequential, Test_sentences_with_double_symbols) {
  // Create data
  std::string in =
      "Hello!! My name is Grisha!! I have two pets: cat,dog,parrot. What is your name?! How are you!? Well...";
  std::vector<char> global_str;
  for (int i = 0; i < in.length(); i++) {
    global_str.push_back(in[i]);
  }
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  kolodkin_g_sentence_count_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out[0], 6);
}
TEST(Sequential, Big_text) {
  // Create data
  std::string in =
      "Otche nash, ize esi na nebeseh! Da svytitsa imya tvoe, da priidet tsarstvo tvoe! Da budet volya tvoya, ako na "
      "nebeseh i na zemle. Hleb nas nasyshnii dazd nam dnes, i ostavi nam dolgi nasha. Yakozhe i my ostavlyaem "
      "dolznikom nashim! I ne vvedi nas vo iskushenie, no izbavi nas ot lukavogo... Amin!";
  std::vector<char> global_str;
  for (int i = 0; i < in.length(); i++) {
    global_str.push_back(in[i]);
  }
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  kolodkin_g_sentence_count_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out[0], 7);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
