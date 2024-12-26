#include <gtest/gtest.h>

#include <vector>

#include "seq/deryabin_m_symbol_frequency/include/ops_seq.hpp"

TEST(deryabin_m_symbol_frequency_seq, test_alphabet) {
  // Create data
  std::vector<std::string> in(1, "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z");
  std::vector<char> in_ch(1, 'A');
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_ch.data()));
  taskDataSeq->inputs_count.emplace_back(in_ch.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  deryabin_m_symbol_frequency_seq::SymbolFrequencyTaskSequential symbol_frequency_TaskSequential(taskDataSeq);
  ASSERT_EQ(symbol_frequency_TaskSequential.validation(), true);
  symbol_frequency_TaskSequential.pre_processing();
  symbol_frequency_TaskSequential.run();
  symbol_frequency_TaskSequential.post_processing();
  ASSERT_EQ(1, out[0]);
}

TEST(deryabin_m_symbol_frequency_seq, test_string_without_symbol) {
  // Create data
  std::vector<std::string> in(1, "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z");
  std::vector<char> in_ch(1, 'a');
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_ch.data()));
  taskDataSeq->inputs_count.emplace_back(in_ch.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  deryabin_m_symbol_frequency_seq::SymbolFrequencyTaskSequential symbol_frequency_TaskSequential(taskDataSeq);
  ASSERT_EQ(symbol_frequency_TaskSequential.validation(), true);
  symbol_frequency_TaskSequential.pre_processing();
  symbol_frequency_TaskSequential.run();
  symbol_frequency_TaskSequential.post_processing();
  ASSERT_EQ(0, out[0]);
}

TEST(deryabin_m_symbol_frequency_seq, test_large_string) {
  // Create data
  std::vector<std::string> in(1, std::string(5000 - 1, 'A') + 'B' + std::string(5000, 'C'));
  std::vector<char> in_ch(1, 'B');
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_ch.data()));
  taskDataSeq->inputs_count.emplace_back(in_ch.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  deryabin_m_symbol_frequency_seq::SymbolFrequencyTaskSequential symbol_frequency_TaskSequential(taskDataSeq);
  ASSERT_EQ(symbol_frequency_TaskSequential.validation(), true);
  symbol_frequency_TaskSequential.pre_processing();
  symbol_frequency_TaskSequential.run();
  symbol_frequency_TaskSequential.post_processing();
  ASSERT_EQ(1, out[0]);
}

TEST(deryabin_m_symbol_frequency_seq, test_empty_string) {
  // Create data
  std::vector<std::string> in(1, std::string());
  std::vector<char> in_ch(1, 'A');
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_ch.data()));
  taskDataSeq->inputs_count.emplace_back(in_ch.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  deryabin_m_symbol_frequency_seq::SymbolFrequencyTaskSequential symbol_frequency_TaskSequential(taskDataSeq);
  ASSERT_EQ(symbol_frequency_TaskSequential.validation(), true);
  symbol_frequency_TaskSequential.pre_processing();
  symbol_frequency_TaskSequential.run();
  symbol_frequency_TaskSequential.post_processing();
  ASSERT_EQ(0, out[0]);
}

TEST(deryabin_m_symbol_frequency_seq, test_arithmetic_string) {
  // Create data
  std::vector<std::string> in(1, "0.49*exp(a-b*b)+ln(cos(a*a))*3");
  std::vector<char> in_ch(1, 'a');
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_ch.data()));
  taskDataSeq->inputs_count.emplace_back(in_ch.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  deryabin_m_symbol_frequency_seq::SymbolFrequencyTaskSequential symbol_frequency_TaskSequential(taskDataSeq);
  ASSERT_EQ(symbol_frequency_TaskSequential.validation(), true);
  symbol_frequency_TaskSequential.pre_processing();
  symbol_frequency_TaskSequential.run();
  symbol_frequency_TaskSequential.post_processing();
  ASSERT_EQ(3, out[0]);
}
