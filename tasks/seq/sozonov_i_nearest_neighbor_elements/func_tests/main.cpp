#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "seq/sozonov_i_nearest_neighbor_elements/include/ops_seq.hpp"

TEST(sozonov_i_nearest_neighbor_elements_seq, test_nearest_neighbor_elements_empty_vector) {
  // Create data
  std::vector<int> in;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());

  // Create Task
  sozonov_i_nearest_neighbor_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(sozonov_i_nearest_neighbor_elements_seq, test_nearest_neighbor_elements_10) {
  const int count = 10;

  // Create data
  std::vector<int> in(count);
  std::iota(in.begin(), in.end(), 0);
  in[0] = 1;
  std::vector<int> out(2, 0);
  std::vector<int> ans(2, 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_nearest_neighbor_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(sozonov_i_nearest_neighbor_elements_seq, test_nearest_neighbor_elements_20) {
  const int count = 20;

  // Create data
  std::vector<int> in(count);
  std::iota(in.begin(), in.end(), 0);
  in[0] = 1;
  std::vector<int> out(2, 0);
  std::vector<int> ans(2, 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_nearest_neighbor_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(sozonov_i_nearest_neighbor_elements_seq, test_nearest_neighbor_elements_50) {
  const int count = 50;

  // Create data
  std::vector<int> in(count);
  std::iota(in.begin(), in.end(), 0);
  in[0] = 1;
  std::vector<int> out(2, 0);
  std::vector<int> ans(2, 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_nearest_neighbor_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(sozonov_i_nearest_neighbor_elements_seq, test_nearest_neighbor_elements_70) {
  const int count = 70;

  // Create data
  std::vector<int> in(count);
  std::iota(in.begin(), in.end(), 0);
  in[0] = 1;
  std::vector<int> out(2, 0);
  std::vector<int> ans(2, 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_nearest_neighbor_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(sozonov_i_nearest_neighbor_elements_seq, test_nearest_neighbor_elements_100) {
  const int count = 100;

  // Create data
  std::vector<int> in(count);
  std::iota(in.begin(), in.end(), 0);
  in[0] = 1;
  std::vector<int> out(2, 0);
  std::vector<int> ans(2, 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sozonov_i_nearest_neighbor_elements_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}