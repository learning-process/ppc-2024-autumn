#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <functional>
#include <random>
#include <vector>

#include "seq/somov_i_bitwise_sorting_batcher_merge/include/ops_seq.hpp"

namespace somov_i_bitwise_sorting_batcher_merge_mpi_seq {

std::vector<double> create_random_vector(int size, double mean = 3.0, double stddev = 300.0) {
  std::normal_distribution<double> norm_dist(mean, stddev);

  std::random_device rand_dev;
  std::mt19937 rand_engine(rand_dev());

  std::vector<double> tmp(size);
  for (int i = 0; i < size; i++) {
    tmp[i] = norm_dist(rand_engine);
  }
  return tmp;
}

}  // namespace somov_i_bitwise_sorting_batcher_merge_mpi_seq

TEST(somov_i_bitwise_sorting_batcher_merge_seq, test_basic_sorting) {
  std::vector<double> in = {3.14, -2.73, 1.41, 0.0, -3.14, 2.73};
  std::vector<double> out(in.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  somov_i_bitwise_sorting_batcher_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 1; i < out.size(); i++) ASSERT_TRUE(out[i - 1] <= out[i]);
}

TEST(somov_i_bitwise_sorting_batcher_merge_seq, test_sorting_10_elements) {
  std::vector<double> in = somov_i_bitwise_sorting_batcher_merge_mpi_seq::create_random_vector(10);
  std::vector<double> out(in.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  somov_i_bitwise_sorting_batcher_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 1; i < out.size(); i++) ASSERT_TRUE(out[i - 1] <= out[i]);
}

TEST(somov_i_bitwise_sorting_batcher_merge_seq, test_sorting_101_elements) {
  std::vector<double> in = somov_i_bitwise_sorting_batcher_merge_mpi_seq::create_random_vector(101);
  std::vector<double> out(in.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  somov_i_bitwise_sorting_batcher_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 1; i < out.size(); i++) ASSERT_TRUE(out[i - 1] <= out[i]);
}

TEST(somov_i_bitwise_sorting_batcher_merge_seq, test_sorting_1007_elements) {
  std::vector<double> in = somov_i_bitwise_sorting_batcher_merge_mpi_seq::create_random_vector(1007);
  std::vector<double> out(in.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  somov_i_bitwise_sorting_batcher_merge_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 1; i < out.size(); i++) ASSERT_TRUE(out[i - 1] <= out[i]);
}
