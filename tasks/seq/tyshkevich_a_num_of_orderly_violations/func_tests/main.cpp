#include <gtest/gtest.h>

#include <vector>

#include "seq/tyshkevich_a_num_of_orderly_violations/include/ops_seq.hpp"

TEST(tyshkevich_a_num_of_orderly_violations_seq_ftest, Test_10) {
  int size = 10;

  // Create data
  std::vector<int> in{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> out(1, 0);
  int solution = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  tyshkevich_a_num_of_orderly_violations_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(solution, out[0]);
}

TEST(tyshkevich_a_num_of_orderly_violations_seq_ftest, Test_1) {
  int size = 1;

  // Create data
  std::vector<int> in{1};
  std::vector<int> out(1, 0);
  int solution = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  tyshkevich_a_num_of_orderly_violations_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(solution, out[0]);
}

TEST(tyshkevich_a_num_of_orderly_violations_seq_ftest, Test_12) {
  int size = 12;

  // Create data
  std::vector<int> in{1, 2, 4, 6, 1, 8, 3, 0, 5, 9, 4, 4};
  std::vector<int> out(1, 0);
  int solution = 4;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  tyshkevich_a_num_of_orderly_violations_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(solution, out[0]);
}

TEST(tyshkevich_a_num_of_orderly_violations_seq_ftest, Test_20) {
  int size = 20;

  // Create data
  std::vector<int> in{1, 2, 4, 6, 1, 8, 3, 0, 5, 9, 4, 2, 4, 6, 1, 8, 3, 4, 5, 7};
  std::vector<int> out(1, 0);
  int solution = 7;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  tyshkevich_a_num_of_orderly_violations_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(solution, out[0]);
}

TEST(tyshkevich_a_num_of_orderly_violations_seq_ftest, Test_50) {
  int size = 50;

  // Create data
  std::vector<int> in{1, 2, 4, 6, 1, 8, 3, 0, 5, 9, 4, 2, 4, 6, 1, 8, 3, 4, 5, 7, 4, 5, 6, 7, 1,
                      2, 5, 4, 6, 2, 4, 6, 2, 1, 6, 8, 4, 5, 6, 7, 8, 9, 3, 6, 7, 8, 2, 3, 2, 3};
  std::vector<int> out(1, 0);
  int solution = 17;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  tyshkevich_a_num_of_orderly_violations_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(solution, out[0]);
}
