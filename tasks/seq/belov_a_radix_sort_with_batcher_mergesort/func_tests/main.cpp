#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/belov_a_radix_sort_with_batcher_mergesort/include/ops_seq.hpp"

using namespace belov_a_radix_batcher_mergesort_seq;

namespace belov_a_radix_batcher_mergesort_seq {
vector<bigint> generate_mixed_values_array(int n) {
  random_device rd;
  mt19937 gen(rd());

  uniform_int_distribution<bigint> small_range(-999LL, 999LL);
  uniform_int_distribution<bigint> medium_range(-10000LL, 10000LL);
  uniform_int_distribution<bigint> large_range(-10000000000LL, 10000000000LL);
  uniform_int_distribution<int> choice(0, 2);

  vector<bigint> arr;
  arr.reserve(n);

  for (int i = 0; i < n; ++i) {
    if (choice(gen) == 0) {
      arr.push_back(small_range(gen));
    } else if (choice(gen) == 1) {
      arr.push_back(medium_range(gen));
    } else {
      arr.push_back(large_range(gen));
    }
  }
  return arr;
}

vector<bigint> generate_int_array(int n) {
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<int> dist(numeric_limits<int>::min(), numeric_limits<int>::max());

  vector<bigint> arr;
  arr.reserve(n);

  for (int i = 0; i < n; ++i) {
    arr.push_back(dist(gen));
  }
  return arr;
}

vector<bigint> generate_bigint_array(int n) {
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<bigint> dist(numeric_limits<bigint>::min() / 2, numeric_limits<bigint>::max() / 2);

  vector<bigint> arr;
  arr.reserve(n);

  for (int i = 0; i < n; ++i) {
    arr.push_back(dist(gen));
  }
  return arr;
}
}  // namespace belov_a_radix_batcher_mergesort_seq

TEST(belov_a_radix_batcher_mergesort_seq, test_random_small_bigintV_vector) {
  int n = 1024;
  vector<bigint> arr = generate_bigint_array(n);

  vector<bigint> expected_solution = arr;
  std::sort(expected_solution.begin(), expected_solution.end());

  shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->inputs_count.emplace_back(arr.size());
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_seq, test_random_medium_bigintV_vector) {
  int n = 32768;
  vector<bigint> arr = generate_bigint_array(n);

  vector<bigint> expected_solution = arr;
  std::sort(expected_solution.begin(), expected_solution.end());

  shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->inputs_count.emplace_back(arr.size());
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_seq, test_random_large_bigintV_vector) {
  int n = 524288;
  vector<bigint> arr = generate_bigint_array(n);

  vector<bigint> expected_solution = arr;
  std::sort(expected_solution.begin(), expected_solution.end());

  shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->inputs_count.emplace_back(arr.size());
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_seq, test_random_small_intV_vector) {
  int n = 1024;
  vector<bigint> arr = generate_int_array(n);

  vector<bigint> expected_solution = arr;
  std::sort(expected_solution.begin(), expected_solution.end());

  shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->inputs_count.emplace_back(arr.size());
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_seq, test_random_medium_intV_vector) {
  int n = 32768;
  vector<bigint> arr = generate_int_array(n);

  vector<bigint> expected_solution = arr;
  std::sort(expected_solution.begin(), expected_solution.end());

  shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->inputs_count.emplace_back(arr.size());
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_seq, test_random_large_intV_vector) {
  int n = 524288;
  vector<bigint> arr = generate_int_array(n);

  vector<bigint> expected_solution = arr;
  std::sort(expected_solution.begin(), expected_solution.end());

  shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->inputs_count.emplace_back(arr.size());
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_seq, test_random_small_mixedV_vector) {
  int n = 2048;
  vector<bigint> arr = generate_mixed_values_array(n);

  vector<bigint> expected_solution = arr;
  std::sort(expected_solution.begin(), expected_solution.end());

  shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->inputs_count.emplace_back(arr.size());
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_seq, test_random_medium_mixedV_vector) {
  int n = 65536;
  vector<bigint> arr = generate_mixed_values_array(n);

  vector<bigint> expected_solution = arr;
  std::sort(expected_solution.begin(), expected_solution.end());

  shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->inputs_count.emplace_back(arr.size());
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_seq, test_random_large_mixedV_vector) {
  int n = 524288;
  vector<bigint> arr = generate_mixed_values_array(n);

  vector<bigint> expected_solution = arr;
  std::sort(expected_solution.begin(), expected_solution.end());

  shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->inputs_count.emplace_back(arr.size());
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_seq, test_predefined_intV_vector) {
  int n = 16;
  vector<bigint> arr = {74685421,  -53749, 2147483647, -1000, -2147483648, 1001, 0,       124,
                        315986930, -123,   42,         -43,   2,           -1,   -999999, 999998};

  vector<bigint> expected_solution = arr;
  std::sort(expected_solution.begin(), expected_solution.end());

  shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->inputs_count.emplace_back(arr.size());
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_seq, test_one_element_input_bigint) {
  int n = 1;
  vector<bigint> arr = {8888};

  vector<bigint> expected_solution = arr;

  shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->inputs_count.emplace_back(arr.size());
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_seq, test_array_size_missmatch) {
  int n = 3;
  vector<bigint> arr = {-53742329, -2147483648, 123265244, 0, 315986930, 42, 2147483647, -853960, 472691};

  shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->inputs_count.emplace_back(arr.size());
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortSequential testTaskSequential(taskDataSeq);

  EXPECT_FALSE(testTaskSequential.validation());
}

TEST(belov_a_radix_batcher_mergesort_seq, test_invalid_inputs_count) {
  int n = 3;
  vector<bigint> arr = {-53742329, -2147483648, 123265244, 0, 315986930, 42, 2147483647, -853960, 472691};

  shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  // taskDataSeq->inputs_count.emplace_back(arr.size());
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortSequential testTaskSequential(taskDataSeq);

  EXPECT_FALSE(testTaskSequential.validation());
}

TEST(belov_a_radix_batcher_mergesort_seq, test_empty_input_validation) {
  int n = 0;
  vector<bigint> arr = {};

  shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->inputs_count.emplace_back(arr.size());
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortSequential testTaskSequential(taskDataSeq);

  EXPECT_FALSE(testTaskSequential.validation());
}

TEST(belov_a_radix_batcher_mergesort_seq, test_empty_output_validation) {
  int n = 3;
  vector<bigint> arr = {789, 654, 231, 0, 123456789, 792012345678, -22475942, -853960, 59227648};

  shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->inputs_count.emplace_back(arr.size());
  taskDataSeq->inputs_count.emplace_back(n);
  // taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  taskDataSeq->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortSequential testTaskSequential(taskDataSeq);

  EXPECT_FALSE(testTaskSequential.validation());
}