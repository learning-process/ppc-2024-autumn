#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/sharamygina_i_most_different_neighbor_elements/include/ops_seq.hpp"

namespace sharamygina_i_most_different_neighbor_elements_seq {
void generator(std::vector<int> &v) {
  std::random_device dev;
  std::mt19937 gen(dev());

  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = -1000 + gen() % 1000;
  }
}
}  // namespace sharamygina_i_most_different_neighbor_elements_seq

TEST(sharamygina_i_most_different_neighbor_elements_seq, wrong_data_1) {
  std::vector<int> in = {};
  std::vector<std::vector<int>> out(1);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  sharamygina_i_most_different_neighbor_elements_seq::most_different_neighbor_elements_seq testTaskSequential(
      taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(sharamygina_i_most_different_neighbor_elements_seq, wrong_data_2) {
  std::vector<int> in(1);
  std::vector<std::vector<int>> out(1);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  sharamygina_i_most_different_neighbor_elements_seq::most_different_neighbor_elements_seq testTaskSequential(
      taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(sharamygina_i_most_different_neighbor_elements_seq, minimum_size_test) {
  std::vector<int> in(2);
  sharamygina_i_most_different_neighbor_elements_seq::generator(in);
  std::vector<int> out(1);
  int ans = abs(in[0] - in[1]);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  sharamygina_i_most_different_neighbor_elements_seq::most_different_neighbor_elements_seq testTaskSequential(
      taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out[0]);
}

TEST(sharamygina_i_most_different_neighbor_elements_seq, random_test) {
  std::vector<int> in = {12, 23, 46, 84, 50, 10000, 12};
  std::vector<int> out(1);
  int ans = 9988;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  sharamygina_i_most_different_neighbor_elements_seq::most_different_neighbor_elements_seq testTaskSequential(
      taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out[0]);
}

TEST(sharamygina_i_most_different_neighbor_elements_seq, ne_no_null_number) {
  std::vector<int> in = {0, 0, 0, 0, 0, 0, 0, 1000000, 0, 0};
  std::vector<int> out(1);
  int ans = 1000000;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  sharamygina_i_most_different_neighbor_elements_seq::most_different_neighbor_elements_seq testTaskSequential(
      taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out[0]);
}

TEST(sharamygina_i_most_different_neighbor_elements_seq, all_eqal_numbers) {
  std::vector<int> in(12, 10);
  std::vector<int> out(1);
  int ans = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  sharamygina_i_most_different_neighbor_elements_seq::most_different_neighbor_elements_seq testTaskSequential(
      taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out[0]);
}

TEST(sharamygina_i_most_different_neighbor_elements_seq, test_with_neg_numbers) {
  std::vector<int> in = {-1000, -100000, -12};
  std::vector<int> out(1);
  int ans = 99988;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  sharamygina_i_most_different_neighbor_elements_seq::most_different_neighbor_elements_seq testTaskSequential(
      taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out[0]);
}

TEST(sharamygina_i_most_different_neighbor_elements_seq, test_with_rand) {
  std::vector<int> in = {1, 10, 12, 34, 87, 90, 5, 15, 10, 30, 22, 101, 77, 89};
  std::vector<int> out(1);
  int ans = 85;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  sharamygina_i_most_different_neighbor_elements_seq::most_different_neighbor_elements_seq testTaskSequential(
      taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out[0]);
}
