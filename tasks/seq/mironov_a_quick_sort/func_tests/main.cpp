#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "seq/mironov_a_quick_sort/include/ops_mpi.hpp"

namespace mironov_a_quick_sort_seq {

std::vector<int> get_random_vector(int sz, int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);

  int mod = max - min + 1;
  if (mod < 0) {
    mod *= -1;
  }
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % mod + min;
  }

  return vec;
}
}  // namespace mironov_a_quick_sort_seq

TEST(mironov_a_quick_sort_seq, Test_sort_1) {
  const int count = 20;
  std::vector<int> gold(count);

  // Create data
  std::vector<int> in(count);
  std::vector<int> out(count);
  for (int i = 0; i < count; ++i) {
    in[i] = count - 1 - i;
    gold[i] = i;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> seqTask = std::make_shared<ppc::core::TaskData>();
  seqTask->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  seqTask->inputs_count.emplace_back(in.size());
  seqTask->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  seqTask->outputs_count.emplace_back(out.size());

  // Create Task
  mironov_a_quick_sort_seq::QuickSortSequential QuickSortSequential(seqTask);
  ASSERT_EQ(QuickSortSequential.validation(), true);
  QuickSortSequential.pre_processing();
  QuickSortSequential.run();
  QuickSortSequential.post_processing();
  ASSERT_EQ(gold, out);
}

TEST(mironov_a_quick_sort_seq, Test_sort_2) {
  const int count = 30000;
  std::vector<int> gold(count);

  // Create data
  std::vector<int> in = mironov_a_quick_sort_seq::get_random_vector(count, 0, 1000000);
  gold = in;
  std::sort(gold.begin(), gold.end());
  std::vector<int> out(count);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> seqTask = std::make_shared<ppc::core::TaskData>();
  seqTask->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  seqTask->inputs_count.emplace_back(in.size());
  seqTask->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  seqTask->outputs_count.emplace_back(out.size());

  // Create Task
  mironov_a_quick_sort_seq::QuickSortSequential QuickSortSequential_(seqTask);
  ASSERT_EQ(QuickSortSequential_.validation(), true);
  QuickSortSequential_.pre_processing();
  QuickSortSequential_.run();
  QuickSortSequential_.post_processing();
  ASSERT_EQ(gold, out);
}

TEST(mironov_a_quick_sort_seq, Test_sort_3) {
  const int count = 5000;
  std::vector<int> gold(count);

  // Create data
  std::vector<int> in = mironov_a_quick_sort_seq::get_random_vector(count, 0, 10);
  gold = in;
  std::sort(gold.begin(), gold.end());
  std::vector<int> out(count);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> seqTask = std::make_shared<ppc::core::TaskData>();
  seqTask->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  seqTask->inputs_count.emplace_back(in.size());
  seqTask->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  seqTask->outputs_count.emplace_back(out.size());

  // Create Task
  mironov_a_quick_sort_seq::QuickSortSequential QuickSortSequential(seqTask);
  ASSERT_EQ(QuickSortSequential.validation(), true);
  QuickSortSequential.pre_processing();
  QuickSortSequential.run();
  QuickSortSequential.post_processing();
  ASSERT_EQ(gold, out);
}

TEST(mironov_a_quick_sort_seq, Test_sort_4) {
  const int count = 1024;
  std::vector<int> gold(count);

  // Create data
  std::vector<int> in = mironov_a_quick_sort_seq::get_random_vector(count, -10000, -1000);
  gold = in;
  std::sort(gold.begin(), gold.end());
  std::vector<int> out(count);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> seqTask = std::make_shared<ppc::core::TaskData>();
  seqTask->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  seqTask->inputs_count.emplace_back(in.size());
  seqTask->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  seqTask->outputs_count.emplace_back(out.size());

  // Create Task
  mironov_a_quick_sort_seq::QuickSortSequential QuickSortSequential(seqTask);
  ASSERT_EQ(QuickSortSequential.validation(), true);
  QuickSortSequential.pre_processing();
  QuickSortSequential.run();
  QuickSortSequential.post_processing();
  ASSERT_EQ(gold, out);
}

TEST(mironov_a_quick_sort_seq, Test_sort_5) {
  const int count = 10;
  std::vector<int> gold(count);

  // Create data
  std::vector<int> in = mironov_a_quick_sort_seq::get_random_vector(count, 100, std::numeric_limits<int>::max());
  gold = in;
  std::sort(gold.begin(), gold.end());
  std::vector<int> out(count);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> seqTask = std::make_shared<ppc::core::TaskData>();
  seqTask->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  seqTask->inputs_count.emplace_back(in.size());
  seqTask->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  seqTask->outputs_count.emplace_back(out.size());

  // Create Task
  mironov_a_quick_sort_seq::QuickSortSequential QuickSortSequential(seqTask);
  ASSERT_EQ(QuickSortSequential.validation(), true);
  QuickSortSequential.pre_processing();
  QuickSortSequential.run();
  QuickSortSequential.post_processing();
  ASSERT_EQ(gold, out);
}

TEST(mironov_a_quick_sort_seq, Test_sort_6) {
  const int count = 1;
  std::vector<int> gold(count);

  // Create data
  std::vector<int> in = mironov_a_quick_sort_seq::get_random_vector(count, -100, 100);
  gold = in;
  std::sort(gold.begin(), gold.end());
  std::vector<int> out(count);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> seqTask = std::make_shared<ppc::core::TaskData>();
  seqTask->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  seqTask->inputs_count.emplace_back(in.size());
  seqTask->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  seqTask->outputs_count.emplace_back(out.size());

  // Create Task
  mironov_a_quick_sort_seq::QuickSortSequential QuickSortSequential(seqTask);
  ASSERT_EQ(QuickSortSequential.validation(), true);
  QuickSortSequential.pre_processing();
  QuickSortSequential.run();
  QuickSortSequential.post_processing();
  ASSERT_EQ(gold, out);
}

TEST(mironov_a_quick_sort_seq, Test_sort_reversed_array) {
  const int count = 1;
  std::vector<int> gold(count);

  // Create data
  std::vector<int> in = mironov_a_quick_sort_seq::get_random_vector(count, -10000, 10000);
  gold = in;
  std::sort(gold.begin(), gold.end());
  std::sort(in.rbegin(), in.rend());
  std::vector<int> out(count);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> seqTask = std::make_shared<ppc::core::TaskData>();
  seqTask->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  seqTask->inputs_count.emplace_back(in.size());
  seqTask->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  seqTask->outputs_count.emplace_back(out.size());

  // Create Task
  mironov_a_quick_sort_seq::QuickSortSequential QuickSortSequential(seqTask);
  ASSERT_EQ(QuickSortSequential.validation(), true);
  QuickSortSequential.pre_processing();
  QuickSortSequential.run();
  QuickSortSequential.post_processing();
  ASSERT_EQ(gold, out);
}

TEST(mironov_a_quick_sort_seq, Test_wrong_input) {
  // Create data
  std::vector<int> in;
  std::vector<int> out;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> seqTask = std::make_shared<ppc::core::TaskData>();
  seqTask->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  seqTask->inputs_count.emplace_back(in.size());
  seqTask->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  seqTask->outputs_count.emplace_back(out.size());

  // Create Task
  mironov_a_quick_sort_seq::QuickSortSequential QuickSortSequential(seqTask);
  ASSERT_EQ(QuickSortSequential.validation(), false);
}
