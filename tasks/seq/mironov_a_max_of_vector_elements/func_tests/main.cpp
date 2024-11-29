#include <gtest/gtest.h>

#include <vector>

#include "seq/mironov_a_max_of_vector_elements/include/ops_seq.hpp"

TEST(mironov_a_max_of_vector_elements_seq, Test_Max_1) {
  const int count = 10000;
  const int gold = 9999;

  // Create data
  std::vector<int> in(count);
  std::vector<int> out(1);
  for (int i = 0; i < count; ++i) {
    in[i] = i;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  mironov_a_max_of_vector_elements_seq::MaxVectorSequential MaxVectorSequential(taskDataSeq);
  ASSERT_EQ(MaxVectorSequential.validation(), true);
  MaxVectorSequential.pre_processing();
  MaxVectorSequential.run();
  MaxVectorSequential.post_processing();
  ASSERT_EQ(gold, out[0]);
}

TEST(mironov_a_max_of_vector_elements_seq, Test_Max_2) {
  const int count = 1;
  const int gold = -100000000;

  // Create data
  std::vector<int> in(count, -100000000);
  std::vector<int> out(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  mironov_a_max_of_vector_elements_seq::MaxVectorSequential MaxVectorSequential(taskDataSeq);
  ASSERT_EQ(MaxVectorSequential.validation(), true);
  MaxVectorSequential.pre_processing();
  MaxVectorSequential.run();
  MaxVectorSequential.post_processing();
  ASSERT_EQ(gold, out[0]);
}

TEST(mironov_a_max_of_vector_elements_seq, Test_Max_3) {
  constexpr int count = 10000000;
  constexpr int start = -7890000;
  constexpr int gold = start + 9 * (count - 1);

  // Create data
  std::vector<int> in(count);
  std::vector<int> out(1);
  for (int i = 0, j = start; i < count; ++i, j += 9) {
    in[i] = j;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  mironov_a_max_of_vector_elements_seq::MaxVectorSequential MaxVectorSequential(taskDataSeq);
  ASSERT_EQ(MaxVectorSequential.validation(), true);
  MaxVectorSequential.pre_processing();
  MaxVectorSequential.run();
  MaxVectorSequential.post_processing();
  ASSERT_EQ(gold, out[0]);
}

TEST(mironov_a_max_of_vector_elements_seq, Test_Max_4) {
  constexpr int count = 10000000;
  constexpr int start = -7890000;
  constexpr int gold = start + 4 * (count - 1);

  // Create data
  std::vector<int> in(count);
  std::vector<int> out(1);
  for (int i = count - 1, j = start; i >= 0; --i, j += 4) {
    in[i] = j;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  mironov_a_max_of_vector_elements_seq::MaxVectorSequential MaxVectorSequential(taskDataSeq);
  ASSERT_EQ(MaxVectorSequential.validation(), true);
  MaxVectorSequential.pre_processing();
  MaxVectorSequential.run();
  MaxVectorSequential.post_processing();
  ASSERT_EQ(gold, out[0]);
}

TEST(mironov_a_max_of_vector_elements_seq, Test_Max_5) {
  const int count = 100;
  const int gold = INT_MAX;

  // Create data
  std::vector<int> in(count, 0);
  std::vector<int> out(1);
  for (int i = 1; i < 100; i += 2) {
    in[i] = INT_MAX;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  mironov_a_max_of_vector_elements_seq::MaxVectorSequential MaxVectorSequential(taskDataSeq);
  ASSERT_EQ(MaxVectorSequential.validation(), true);
  MaxVectorSequential.pre_processing();
  MaxVectorSequential.run();
  MaxVectorSequential.post_processing();
  ASSERT_EQ(gold, out[0]);
}

TEST(mironov_a_max_of_vector_elements_seq, Wrong_Input_1) {
  // Create data
  std::vector<int> in;
  std::vector<int> out(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  mironov_a_max_of_vector_elements_seq::MaxVectorSequential MaxVectorSequential(taskDataSeq);
  ASSERT_EQ(MaxVectorSequential.validation(), false);
}

TEST(mironov_a_max_of_vector_elements_seq, Wrong_Input_2) {
  // Create data
  std::vector<int> in(3, 5);
  std::vector<int> out;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  mironov_a_max_of_vector_elements_seq::MaxVectorSequential MaxVectorSequential(taskDataSeq);
  ASSERT_EQ(MaxVectorSequential.validation(), false);
}
