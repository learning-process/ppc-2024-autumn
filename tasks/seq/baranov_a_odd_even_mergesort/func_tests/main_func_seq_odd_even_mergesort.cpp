#include <gtest/gtest.h>

#include "seq/baranov_a_odd_even_mergesort/include/header_seq_odd_even.hpp"
TEST(baranov_a_qsort_odd_even_merge_seq, Test_odd_even_sort_0_int) {
  const int N = 0;
  // Create data
  std::vector<int> arr(N);
  std::vector<int> out(N);
  std::random_device rd;
  std::default_random_engine reng(rd());
  std::uniform_int_distribution<int> dist(0, N);
  std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  data_seq->inputs_count.emplace_back(arr.size());
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(1);
  baranov_a_qsort_odd_even_merge_seq::odd_even_mergesort_seq<int> test1(data_seq);
  ASSERT_EQ(test1.validation(), false);
}

TEST(baranov_a_qsort_odd_even_merge_seq, Test_odd_even_sort_1000_int) {
  const int N = 1000;
  // Create data
  std::vector<int> arr(N);
  std::vector<int> out(N);
  std::random_device rd;
  std::default_random_engine reng(rd());
  std::uniform_int_distribution<int> dist(0, N);
  std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  data_seq->inputs_count.emplace_back(arr.size());
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(1);
  baranov_a_qsort_odd_even_merge_seq::odd_even_mergesort_seq<int> test1(data_seq);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  std::sort(arr.begin(), arr.end());
  ASSERT_EQ(arr, out);
}

TEST(baranov_a_qsort_odd_even_merge_seq, Test_odd_even_sort_4096_int) {
  const int N = 4096;
  // Create data
  std::vector<int> arr(N);
  std::vector<int> out(N);
  std::random_device rd;
  std::default_random_engine reng(rd());
  std::uniform_int_distribution<unsigned> dist(0, N);
  std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  data_seq->inputs_count.emplace_back(arr.size());
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(1);
  baranov_a_qsort_odd_even_merge_seq::odd_even_mergesort_seq<int> test1(data_seq);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  std::sort(arr.begin(), arr.end());
  ASSERT_EQ(arr, out);
}

TEST(baranov_a_qsort_odd_even_merge_seq, Test_odd_even_sort_1000_double) {
  const int N = 1000;
  // Create data
  std::vector<double> arr(N);

  std::random_device rd;
  std::default_random_engine reng(rd());
  std::uniform_real_distribution<double> dist(-100.0, 100.0);
  std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
  std::vector<double> out(arr.size());
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  data_seq->inputs_count.emplace_back(arr.size());
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(1);
  baranov_a_qsort_odd_even_merge_seq::odd_even_mergesort_seq<double> test1(data_seq);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  std::sort(arr.begin(), arr.end());
  ASSERT_EQ(arr, out);
}

TEST(baranov_a_qsort_odd_even_merge_seq, Test_odd_even_sort_10000_double) {
  const int N = 10000;
  // Create data
  std::vector<double> arr(N);
  std::vector<double> out(N);
  std::random_device rd;
  std::default_random_engine reng(rd());
  std::uniform_real_distribution<double> dist(0, N);
  std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  data_seq->inputs_count.emplace_back(arr.size());
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(1);
  baranov_a_qsort_odd_even_merge_seq::odd_even_mergesort_seq<double> test1(data_seq);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  std::sort(arr.begin(), arr.end());
  ASSERT_EQ(arr, out);
}

TEST(baranov_a_qsort_odd_even_merge_seq, Test_odd_even_sort_0_uint) {
  const int N = 0;
  // Create data
  std::vector<unsigned> arr(N);
  std::vector<unsigned> out(N);
  std::random_device rd;
  std::default_random_engine reng(rd());
  std::uniform_int_distribution<unsigned> dist(0, N);
  std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  data_seq->inputs_count.emplace_back(arr.size());
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(1);
  baranov_a_qsort_odd_even_merge_seq::odd_even_mergesort_seq<unsigned> test1(data_seq);
  ASSERT_EQ(test1.validation(), false);
}

TEST(baranov_a_qsort_odd_even_merge_seq, Test_odd_even_sort_2000_uint) {
  const int N = 2000;
  // Create data
  std::vector<unsigned> arr(N);
  std::vector<unsigned> out(N);
  std::random_device rd;
  std::default_random_engine reng(rd());
  std::uniform_int_distribution<unsigned> dist(0, N);
  std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  data_seq->inputs_count.emplace_back(arr.size());
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(1);
  baranov_a_qsort_odd_even_merge_seq::odd_even_mergesort_seq<unsigned> test1(data_seq);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  std::sort(arr.begin(), arr.end());
  ASSERT_EQ(arr, out);
}

TEST(baranov_a_qsort_odd_even_merge_seq, Test_odd_even_sort_8000_uint) {
  const int N = 8000;
  // Create data
  std::vector<unsigned> arr(N);
  std::vector<unsigned> out(N);
  std::random_device rd;
  std::default_random_engine reng(rd());
  std::uniform_int_distribution<unsigned> dist(0, N);
  std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  data_seq->inputs_count.emplace_back(arr.size());
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(1);
  baranov_a_qsort_odd_even_merge_seq::odd_even_mergesort_seq<unsigned> test1(data_seq);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  std::sort(arr.begin(), arr.end());
  ASSERT_EQ(arr, out);
}