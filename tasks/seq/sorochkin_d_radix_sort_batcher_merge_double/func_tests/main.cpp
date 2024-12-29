#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "../include/ops_seq.hpp"

static void rsbmd_test(std::vector<double> &&in) {
  // Create data
  std::vector<double> out(in.size(), 0.);

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  // in
  taskDataSeq->inputs = {reinterpret_cast<uint8_t *>(in.data())};
  taskDataSeq->inputs_count = {static_cast<uint32_t>(in.size())};
  // out
  taskDataSeq->outputs = {reinterpret_cast<uint8_t *>(out.data())};
  taskDataSeq->outputs_count = {static_cast<uint32_t>(out.size())};

  // Create Task
  sorochkin_d_radix_sort_batcher_merge_double_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  std::sort(in.begin(), in.end());
  ASSERT_EQ(out, in);
}

static void rsbmd_random_test(size_t sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<double> vec(sz);
  for (size_t i = 0; i < sz; i++) {
    vec[i] = -50 + (gen() / 10.);
  }

  rsbmd_test(std::move(vec));
}

TEST(sorochkin_d_radix_sort_batcher_merge_double_seq, Test_Determined_0) { rsbmd_test({}); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_seq, Test_Determined_Diff) { rsbmd_test({4e5 + 8, 2e-4 - 4, 1e-10}); }

TEST(sorochkin_d_radix_sort_batcher_merge_double_seq, Test_1) { rsbmd_random_test(1); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_seq, Test_2) { rsbmd_random_test(2); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_seq, Test_3) { rsbmd_random_test(3); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_seq, Test_4) { rsbmd_random_test(4); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_seq, Test_5) { rsbmd_random_test(5); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_seq, Test_6) { rsbmd_random_test(6); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_seq, Test_7) { rsbmd_random_test(7); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_seq, Test_8) { rsbmd_random_test(8); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_seq, Test_9) { rsbmd_random_test(9); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_seq, Test_10) { rsbmd_random_test(10); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_seq, Test_13) { rsbmd_random_test(13); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_seq, Test_17) { rsbmd_random_test(17); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_seq, Test_23) { rsbmd_random_test(23); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_seq, Test_31) { rsbmd_random_test(31); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_seq, Test_63) { rsbmd_random_test(63); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_seq, Test_64) { rsbmd_random_test(64); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_seq, Test_65) { rsbmd_random_test(65); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_seq, Test_127) { rsbmd_random_test(127); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_seq, Test_128) { rsbmd_random_test(128); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_seq, Test_129) { rsbmd_random_test(129); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_seq, Test_255) { rsbmd_random_test(255); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_seq, Test_256) { rsbmd_random_test(256); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_seq, Test_257) { rsbmd_random_test(256); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_seq, Test_1000) { rsbmd_random_test(1000); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_seq, Test_1001) { rsbmd_random_test(1001); }