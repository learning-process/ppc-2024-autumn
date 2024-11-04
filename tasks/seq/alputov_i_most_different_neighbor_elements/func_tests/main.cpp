#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/alputov_i_most_different_neighbor_elements/include/ops_seq.hpp"

namespace alputov_i_most_different_neighbor_elements_seq {
std::vector<int> generator(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());

  std::vector<int> ans(sz);
  for (int i = 0; i < sz; ++i) {
    ans[i] = gen() % 1000;
    int x = gen() % 2;
    if (x == 0) ans[i] *= -1;
  }

  return ans;
}
}  // namespace alputov_i_most_different_neighbor_elements_seq

TEST(alputov_i_most_different_neighbor_elements_seq, EmptyInput_ReturnsFalse) {
  std::vector<int> in = {};
  std::vector<std::vector<int>> out(1);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  alputov_i_most_different_neighbor_elements_seq::most_different_neighbor_elements_seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(alputov_i_most_different_neighbor_elements_seq, InputSizeTwo_ReturnsCorrectPair) {
  std::vector<int> in = alputov_i_most_different_neighbor_elements_seq::generator(2);
  std::vector<std::pair<int, int>> out(1);
  std::pair<int, int> ans = {std::min(in[0], in[1]), std::max(in[0], in[1])};

  std::cout << in[0] << " " << in[1] << std::endl;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  alputov_i_most_different_neighbor_elements_seq::most_different_neighbor_elements_seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out[0]);
}

TEST(alputov_i_most_different_neighbor_elements_seq, SequentialInput_ReturnsFirstTwoElements) {
  std::vector<int> in = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<std::pair<int, int>> out(1);
  std::pair<int, int> ans = {1, 2};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  alputov_i_most_different_neighbor_elements_seq::most_different_neighbor_elements_seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out[0]);
}

TEST(alputov_i_most_different_neighbor_elements_seq, MostlyZerosInput_ReturnsZeroAndLargest) {
  std::vector<int> in = {0, 0, 0, 0, 0, 0, 0, 0, 0, 12};
  std::vector<std::pair<int, int>> out(1);
  std::pair<int, int> ans = {0, 12};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  alputov_i_most_different_neighbor_elements_seq::most_different_neighbor_elements_seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out[0]);
}

TEST(alputov_i_most_different_neighbor_elements_seq, AllZerosInput_ReturnsZeroZero) {
  std::vector<int> in(100, 0);
  std::vector<std::pair<int, int>> out(1);
  std::pair<int, int> ans = {0, 0};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  alputov_i_most_different_neighbor_elements_seq::most_different_neighbor_elements_seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out[0]);
}

TEST(alputov_i_most_different_neighbor_elements_seq, CloseNegativeNumbers_ReturnsCorrectPair) {
  std::vector<int> in = {-1, -2, -3, -4, -1000};
  std::vector<std::pair<int, int>> out(1);
  std::pair<int, int> ans = {-1000, -4};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  alputov_i_most_different_neighbor_elements_seq::most_different_neighbor_elements_seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out[0]);
}