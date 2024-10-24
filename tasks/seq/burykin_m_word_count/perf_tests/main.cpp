#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/burykin_m_word_count/include/ops_seq.hpp"

TEST(BurykinWordCountTest, EmptyString) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  auto task = std::make_shared<burykin_m_word_count::TestTaskSequential>(taskData);
  auto perf = std::make_shared<ppc::core::Perf>(task);

  std::string input;
  int output = 0;
  taskData->inputs = {reinterpret_cast<uint8_t*>(input.data())};
  taskData->inputs_count = {static_cast<unsigned int>(input.size())};
  taskData->outputs = {reinterpret_cast<uint8_t*>(&output)};
  taskData->outputs_count = {1};

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  perf->pipeline_run(perfAttr, perfResults);

  EXPECT_EQ(output, 0);
}

TEST(BurykinWordCountTest, SingleWord) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  auto task = std::make_shared<burykin_m_word_count::TestTaskSequential>(taskData);
  auto perf = std::make_shared<ppc::core::Perf>(task);

  std::string input = "Hello";
  int output = 0;
  taskData->inputs = {reinterpret_cast<uint8_t*>(input.data())};
  taskData->inputs_count = {static_cast<unsigned int>(input.size())};
  taskData->outputs = {reinterpret_cast<uint8_t*>(&output)};
  taskData->outputs_count = {1};

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  perf->pipeline_run(perfAttr, perfResults);

  EXPECT_EQ(output, 1);
}

TEST(BurykinWordCountTest, MultipleWords) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  auto task = std::make_shared<burykin_m_word_count::TestTaskSequential>(taskData);
  auto perf = std::make_shared<ppc::core::Perf>(task);

  std::string input = "This is a test sentence.";
  int output = 0;
  taskData->inputs = {reinterpret_cast<uint8_t*>(input.data())};
  taskData->inputs_count = {static_cast<unsigned int>(input.size())};
  taskData->outputs = {reinterpret_cast<uint8_t*>(&output)};
  taskData->outputs_count = {1};

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  perf->pipeline_run(perfAttr, perfResults);

  EXPECT_EQ(output, 5);
}

TEST(BurykinWordCountTest, WordsWithApostrophes) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  auto task = std::make_shared<burykin_m_word_count::TestTaskSequential>(taskData);
  auto perf = std::make_shared<ppc::core::Perf>(task);

  std::string input = "It's a beautiful day, isn't it?";
  int output = 0;
  taskData->inputs = {reinterpret_cast<uint8_t*>(input.data())};
  taskData->inputs_count = {static_cast<unsigned int>(input.size())};
  taskData->outputs = {reinterpret_cast<uint8_t*>(&output)};
  taskData->outputs_count = {1};

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  perf->pipeline_run(perfAttr, perfResults);

  EXPECT_EQ(output, 6);
}

TEST(BurykinWordCountTest, InvalidInput) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  auto task = std::make_shared<burykin_m_word_count::TestTaskSequential>(taskData);

  std::string input = "Test";
  int output = 0;
  taskData->inputs = {reinterpret_cast<uint8_t*>(input.data())};
  taskData->inputs_count = {0};  // Устанавливаем некорректный размер входных данных
  taskData->outputs = {reinterpret_cast<uint8_t*>(&output)};
  taskData->outputs_count = {1};

  EXPECT_FALSE(task->validation());
}
