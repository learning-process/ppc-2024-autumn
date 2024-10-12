// Copyright 2024 Kabalova Valeria
#include <gtest/gtest.h>

#include <vector>

#include "seq/kabalova_v_count_symbols/include/count_symbols.hpp"

TEST(kabalova_v_count_symbols_seq, EmptyString) {
  std::string string = "";
  int answer = 0;
  std::vector<char> str = kabalova_v_count_symbols_seq::fromStringToChar(string);
  // Create data
  std::vector<std::vector<char>> in(1, str);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  kabalova_v_count_symbols_seq::Task1Seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(answer, out[0]);
}

TEST(kabalova_v_count_symbols_seq, OneSymbolStringNotLetter) {
  std::string string = "8";
  int answer = 0;
  std::vector<char> str = kabalova_v_count_symbols_seq::fromStringToChar(string);
  // Create data
  std::vector<std::vector<char>> in(1, str);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  kabalova_v_count_symbols_seq::Task1Seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(answer, out[0]);
}

TEST(kabalova_v_count_symbols_seq, OneSymbolStringLetter) {
  std::string string = "f";
  int answer = 1;
  std::vector<char> str = kabalova_v_count_symbols_seq::fromStringToChar(string);
  // Create data
  std::vector<std::vector<char>> in(1, str);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  kabalova_v_count_symbols_seq::Task1Seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(answer, out[0]);
}

TEST(kabalova_v_count_symbols_seq, string1) {
  std::string string = "%^&@&^*@#$&^*";
  int answer = 0;
  std::vector<char> str = kabalova_v_count_symbols_seq::fromStringToChar(string);
  // Create data
  std::vector<std::vector<char>> in(1, str);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  // Create Task
  kabalova_v_count_symbols_seq::Task1Seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(answer, out[0]);
}



