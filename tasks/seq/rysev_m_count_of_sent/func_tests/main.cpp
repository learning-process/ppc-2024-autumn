// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>
#include <string>

#include "seq/rysev_m_count_of_sent/include/ops_seq.hpp"

TEST(rysev_m_count_of_sent_seq, simple_test) {
  //create data
  std::string str = "The cake is a lie.";
  std::vector<int> out(1, 0);

  //create taskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
  taskDataSeq->inputs_count.emplace_back(str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  //create task
  rysev_m_count_of_sent_seq::SentCountSequential counter(taskDataSeq);
  ASSERT_EQ(counter.validation(), true);
  counter.pre_processing();
  counter.run();
  counter.post_processing();
  ASSERT_EQ(out[0], 1);
}

TEST(rysev_m_count_of_sent_seq, empty_string) {
  // create data
  std::string str = "";
  std::vector<int> out(1, 0);

  // create taskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
  taskDataSeq->inputs_count.emplace_back(str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // create task
  rysev_m_count_of_sent_seq::SentCountSequential counter(taskDataSeq);
  ASSERT_EQ(counter.validation(), true);
  counter.pre_processing();
  counter.run();
  counter.post_processing();
  ASSERT_EQ(out[0], 0);
}

TEST(rysev_m_count_of_sent_seq, text_without_end_symbol) {
  // create data
  std::string str = "Bring me a bucket, and I'll show you a bucket";
  std::vector<int> out(1, 0);

  // create taskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
  taskDataSeq->inputs_count.emplace_back(str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // create task
  rysev_m_count_of_sent_seq::SentCountSequential counter(taskDataSeq);
  ASSERT_EQ(counter.validation(), true);
  counter.pre_processing();
  counter.run();
  counter.post_processing();
  ASSERT_EQ(out[0], 1);
}

TEST(rysev_m_count_of_sent_seq, text_with_double_or_more_end_symbols) {
  // create data
  std::string str = "Who will you choose: them or us?! Us or them?!!...";
  std::vector<int> out(1, 0);

  // create taskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
  taskDataSeq->inputs_count.emplace_back(str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // create task
  rysev_m_count_of_sent_seq::SentCountSequential counter(taskDataSeq);
  ASSERT_EQ(counter.validation(), true);
  counter.pre_processing();
  counter.run();
  counter.post_processing();
  ASSERT_EQ(out[0], 2);
}

TEST(rysev_m_count_of_sent_seq, sample_test_number_two) {
  // create data
  std::string str = "We both said a lot of things that you'll regret. But I think we can put our differences behind us. For the sake of science. You're a monster.";
  std::vector<int> out(1, 0);

  // create taskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
  taskDataSeq->inputs_count.emplace_back(str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // create task
  rysev_m_count_of_sent_seq::SentCountSequential counter(taskDataSeq);
  ASSERT_EQ(counter.validation(), true);
  counter.pre_processing();
  counter.run();
  counter.post_processing();
  ASSERT_EQ(out[0], 4);
}
//o_O