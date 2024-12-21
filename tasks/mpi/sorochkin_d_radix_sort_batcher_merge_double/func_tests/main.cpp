#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <random>
#include <vector>

#include "../include/ops_mpi.hpp"

static void rsbmd_test(std::vector<double> &&in) {
  boost::mpi::communicator world;

  // Create data
  std::vector<double> out;

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    out.resize(in.size());
    // in
    taskDataSeq->inputs = {reinterpret_cast<uint8_t *>(in.data())};
    taskDataSeq->inputs_count = {static_cast<uint32_t>(in.size())};
    // out
    taskDataSeq->outputs = {reinterpret_cast<uint8_t *>(out.data())};
    taskDataSeq->outputs_count = {static_cast<uint32_t>(out.size())};
  }

  // Create Task
  sorochkin_d_radix_sort_batcher_merge_double_mpi::TestMPITaskParallel testTaskParallel(taskDataSeq);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::sort(in.begin(), in.end());
    ASSERT_EQ(out, in);
  }
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

TEST(sorochkin_d_radix_sort_batcher_merge_double_mpi, Test_0) { rsbmd_test({}); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_mpi, Test_2) { rsbmd_random_test(2); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_mpi, Test_4) { rsbmd_random_test(4); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_mpi, Test_8) { rsbmd_random_test(8); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_mpi, Test_16) { rsbmd_random_test(16); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_mpi, Test_32) { rsbmd_random_test(32); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_mpi, Test_64) { rsbmd_random_test(64); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_mpi, Test_128) { rsbmd_random_test(128); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_mpi, Test_256) { rsbmd_random_test(256); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_mpi, Test_512) { rsbmd_random_test(512); }
TEST(sorochkin_d_radix_sort_batcher_merge_double_mpi, Test_1024) { rsbmd_random_test(1024); }