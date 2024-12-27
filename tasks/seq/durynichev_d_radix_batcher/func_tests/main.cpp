#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "seq/durynichev_d_radix_batcher/include/ops_seq.hpp"

namespace durynichev_d_radix_batcher_seq {

std::vector<double> generateRandomData(int array_size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dist(-1000, 1000);

  std::vector<double> arr(array_size);
  for (size_t i = 0; i < arr.size(); i++) {
    arr[i] = dist(gen);
  }
  return arr;
}

void testTemplate(const std::vector<double>& vec) {
  std::vector<double> arr = vec;
  std::vector<double> result_data(arr.size());

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));

  taskDataSeq->inputs_count.emplace_back(arr.size());

  auto taskSequential = std::make_shared<RadixBatcher>(taskDataSeq);

  if (taskSequential->validation()) {
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();

    std::sort(arr.begin(), arr.end());
    EXPECT_EQ(arr, result_data);
  }
}

}  // namespace durynichev_d_radix_batcher_seq

TEST(durynichev_d_radix_batcher_seq, sort_test_empty) {
  std::vector<double> empty_vector;

  EXPECT_NO_THROW(durynichev_d_radix_batcher_seq::radixSortDouble(empty_vector.begin(), empty_vector.end()));
  EXPECT_TRUE(empty_vector.empty());
}

TEST(durynichev_d_radix_batcher_seq, sort_test_right_work) {
  std::vector<double> data = {3.1, 2.4, 5.6, 1.2, 8.9, 7.3, 4.5, 6.7};
  std::vector<double> expected = {1.2, 2.4, 3.1, 4.5, 5.6, 6.7, 7.3, 8.9};

  durynichev_d_radix_batcher_seq::radixSortDouble(data.begin(), data.end());
  EXPECT_EQ(data, expected);
}

TEST(durynichev_d_radix_batcher_seq, positive_numbers_size_8) {
  durynichev_d_radix_batcher_seq::testTemplate({2.0, 4.2, 1.3, 5.7, 0.1, 8.9, 7.4, 6.1});
}

TEST(durynichev_d_radix_batcher_seq, negative_numbers_size_8) {
  durynichev_d_radix_batcher_seq::testTemplate({-2.0, -4.2, -1.3, -5.7, -0.1, -8.9, -7.4, -6.1});
}

TEST(durynichev_d_radix_batcher_seq, mixed_numbers_size_8) {
  durynichev_d_radix_batcher_seq::testTemplate({-2.0, 4.2, -1.3, 5.7, 0.1, -8.9, 7.4, -6.1});
}

TEST(durynichev_d_radix_batcher_seq, fractional_numbers_size_4) {
  durynichev_d_radix_batcher_seq::testTemplate({0.2, 0.002, 0.0002, 0.02});
}

TEST(durynichev_d_radix_batcher_seq, large_numbers_size_8) {
  durynichev_d_radix_batcher_seq::testTemplate({2e11, 4e16, 1e13, 5e19, 2e10, 8e21, 7e12, 6e15});
}

TEST(durynichev_d_radix_batcher_seq, small_numbers_size_8) {
  durynichev_d_radix_batcher_seq::testTemplate({2e-11, 4e-16, 1e-13, 5e-19, 2e-10, 8e-21, 7e-12, 6e-15});
}

TEST(durynichev_d_radix_batcher_seq, numbers_with_infinity_size_4) {
  durynichev_d_radix_batcher_seq::testTemplate(
      {2.0, -2.0, std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()});
}

TEST(durynichev_d_radix_batcher_seq, identical_numbers_size_4) {
  durynichev_d_radix_batcher_seq::testTemplate({3.3, 3.3, 3.3, 3.3});
}

TEST(durynichev_d_radix_batcher_seq, empty_vector) { durynichev_d_radix_batcher_seq::testTemplate({}); }

TEST(durynichev_d_radix_batcher_seq, single_element) { durynichev_d_radix_batcher_seq::testTemplate({24.0}); }

TEST(durynichev_d_radix_batcher_seq, numbers_close_to_zero_size_4) {
  durynichev_d_radix_batcher_seq::testTemplate({-0.000002, 0.0, 0.000002, 0.000003});
}

TEST(durynichev_d_radix_batcher_seq, very_large_and_very_small_numbers_size_4) {
  durynichev_d_radix_batcher_seq::testTemplate({2e19, 2e-19, -2e19, -2e-19});
}

TEST(durynichev_d_radix_batcher_seq, random_vector_size_1) {
  auto vec = durynichev_d_radix_batcher_seq::generateRandomData(1);
  durynichev_d_radix_batcher_seq::testTemplate(vec);
}

TEST(durynichev_d_radix_batcher_seq, random_vector_size_2) {
  auto vec = durynichev_d_radix_batcher_seq::generateRandomData(2);
  durynichev_d_radix_batcher_seq::testTemplate(vec);
}

TEST(durynichev_d_radix_batcher_seq, random_vector_size_8) {
  auto vec = durynichev_d_radix_batcher_seq::generateRandomData(8);
  durynichev_d_radix_batcher_seq::testTemplate(vec);
}

TEST(durynichev_d_radix_batcher_seq, random_vector_size_64) {
  auto vec = durynichev_d_radix_batcher_seq::generateRandomData(64);
  durynichev_d_radix_batcher_seq::testTemplate(vec);
}

TEST(durynichev_d_radix_batcher_seq, random_vector_size_512) {
  auto vec = durynichev_d_radix_batcher_seq::generateRandomData(512);
  durynichev_d_radix_batcher_seq::testTemplate(vec);
}
