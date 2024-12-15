#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "seq/mironov_a_quick_sort/include/ops_mpi.hpp"

namespace mironov_a_quick_sort_seq {

std::vector<int> get_random_vector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);

  const int mod = 1000000;
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % mod;
  }

  return vec;
}
}  // namespace mironov_a_quick_sort_seq

TEST(mironov_a_quick_sort_seq, Test_sort_1) {
  const int count = 10000;
  std::vector<int> gold(count);

  // Create data
  std::vector<int> in(count);
  std::vector<int> out(count);
  for (int i = 0; i < count; ++i) {
    in[i] = count - 1 - i;
    gold[i] = i;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  mironov_a_quick_sort_seq::QuickSortSequential QuickSortSequential(taskDataSeq);
  ASSERT_EQ(QuickSortSequential.validation(), true);
  QuickSortSequential.pre_processing();
  QuickSortSequential.run();
  QuickSortSequential.post_processing();
  ASSERT_EQ(gold, out);
}

TEST(mironov_a_quick_sort_seq, Test_sort_2) {
  const int count = 500;
  std::vector<int> gold(count);

  // Create data
  std::vector<int> in = mironov_a_quick_sort_seq::get_random_vector(count);
  gold = in;
  std::sort(gold.begin(), gold.end());
  std::vector<int> out(count);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  mironov_a_quick_sort_seq::QuickSortSequential QuickSortSequential_(taskDataSeq);
  ASSERT_EQ(QuickSortSequential_.validation(), true);
  QuickSortSequential_.pre_processing();
  QuickSortSequential_.run();
  QuickSortSequential_.post_processing();
  ASSERT_EQ(gold, out);
}

TEST(mironov_a_quick_sort_seq, Test_sort_3) {
  const int count = 400;
  std::vector<int> gold(count);

  // Create data
  std::vector<int> in = mironov_a_quick_sort_seq::get_random_vector(count);
  gold = in;
  std::sort(gold.begin(), gold.end());
  std::vector<int> out(count);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  mironov_a_quick_sort_seq::QuickSortSequential QuickSortSequential(taskDataSeq);
  ASSERT_EQ(QuickSortSequential.validation(), true);
  QuickSortSequential.pre_processing();
  QuickSortSequential.run();
  QuickSortSequential.post_processing();
  ASSERT_EQ(gold, out);
}
