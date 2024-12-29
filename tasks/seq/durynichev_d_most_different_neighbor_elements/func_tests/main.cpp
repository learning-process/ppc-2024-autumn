#include <gtest/gtest.h>

#include "seq/durynichev_d_most_different_neighbor_elements/include/ops_seq.hpp"

TEST(durynichev_d_most_different_neighbor_elements_seq, functionality_check) {
  // Create data
  std::vector<int> in{1, 5, 2, 10, 3};
  std::vector<int> out{0, 0};
  std::vector<int> want{2, 10};

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  durynichev_d_most_different_neighbor_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(want, out);
}

TEST(durynichev_d_most_different_neighbor_elements_seq, all_elements_are_equal) {
  // Create data
  std::vector<int> in{1, 1, 1, 1, 1, 1};
  std::vector<int> out{0, 0};
  std::vector<int> want{1, 1};

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  durynichev_d_most_different_neighbor_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(want, out);
}

TEST(durynichev_d_most_different_neighbor_elements_seq, check_double_number) {
  // Create data
  std::vector<int> in{-10, 10};
  std::vector<int> out{0, 0};
  std::vector<int> want{-10, 10};

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  durynichev_d_most_different_neighbor_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(want, out);
}

TEST(durynichev_d_most_different_neighbor_elements_seq, check_one_number) {
  // Create data
  std::vector<int> in{1};
  std::vector<int> out{0, 0};

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  durynichev_d_most_different_neighbor_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(durynichev_d_most_different_neighbor_elements_seq, check_validate_func) {
  // Create data
  std::vector<int> in{1, 5, 2, 10, 3};
  std::vector<int> out{0};

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  durynichev_d_most_different_neighbor_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}