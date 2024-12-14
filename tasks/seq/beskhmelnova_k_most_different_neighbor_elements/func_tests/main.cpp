#include <gtest/gtest.h>

#include "seq/beskhmelnova_k_most_different_neighbor_elements/src/seq.cpp"

TEST(beskhmelnova_k_most_different_neighbor_elements_seq, Test_vector_int_100) {
  const int count = 100;

  // Create data
  std::vector<int> in(count);
  std::vector<int> out(2);

  in = beskhmelnova_k_most_different_neighbor_elements_seq::getRandomVector<int>(count);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  beskhmelnova_k_most_different_neighbor_elements_seq::TestTaskSequential<int> testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  int index = testTaskSequential.position_of_first_neighbour_seq(in);
  ASSERT_EQ(in[index], out[0]);
  ASSERT_EQ(in[index + 1], out[1]);
}

TEST(beskhmelnova_k_most_different_neighbor_elements_seq, Test_vector_int_10000) {
  const int count = 10000;

  // Create data
  std::vector<int> in(count, 5);
  std::vector<int> out(2);

  in = beskhmelnova_k_most_different_neighbor_elements_seq::getRandomVector<int>(count);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  beskhmelnova_k_most_different_neighbor_elements_seq::TestTaskSequential<int> testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  int index = testTaskSequential.position_of_first_neighbour_seq(in);
  ASSERT_EQ(in[index], out[0]);
  ASSERT_EQ(in[index + 1], out[1]);
}

TEST(beskhmelnova_k_most_different_neighbor_elements_seq, Test_vector_int_100_equal_elements) {
  const int count = 1000;
  const int elem = 7;

  // Create data
  std::vector<int> in(count, elem);
  std::vector<int> out(2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  beskhmelnova_k_most_different_neighbor_elements_seq::TestTaskSequential<int> testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(elem, out[0]);
  ASSERT_EQ(elem, out[1]);
}

TEST(beskhmelnova_k_most_different_neighbor_elements_seq, Test_1_size_vector_int) {
  const int count = 1;

  // Create data
  std::vector<int> in(count);
  std::vector<int> out(2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  beskhmelnova_k_most_different_neighbor_elements_seq::TestTaskSequential<int> testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(-1, out[0]);
  ASSERT_EQ(-1, out[1]);
}

TEST(beskhmelnova_k_most_different_neighbor_elements_seq, Test_0_size_vector_int) {
  const int count = 0;

  // Create data
  std::vector<int> in(count);
  std::vector<int> out(2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  beskhmelnova_k_most_different_neighbor_elements_seq::TestTaskSequential<int> testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(-1, out[0]);
  ASSERT_EQ(-1, out[1]);
}

TEST(beskhmelnova_k_most_different_neighbor_elements_seq, Test_2_size_vector_int) {
  const int count = 2;

  // Create data
  std::vector<int> in(count);
  std::vector<int> out(2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  beskhmelnova_k_most_different_neighbor_elements_seq::TestTaskSequential<int> testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(in[0], out[0]);
  ASSERT_EQ(in[1], out[1]);
}
