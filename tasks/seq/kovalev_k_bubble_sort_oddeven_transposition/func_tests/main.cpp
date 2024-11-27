#include <gtest/gtest.h>

#include <algorithm>

#include "seq/kovalev_k_bubble_sort_oddeven_transposition/include/header.hpp"

TEST(kovalev_k_bubble_sort_oddeven_transposition_seq, zero_length) {
  std::vector<int> in, out;
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(in.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  kovalev_k_bubble_sort_oddeven_transposition_seq::BubbleSortOddEvenTransposition<int> tmpTaskSeq(taskSeq);
  ASSERT_FALSE(tmpTaskSeq.validation());
}

TEST(kovalev_k_bubble_sort_oddeven_transposition_seq, Test_No_viol_10_int) {
  const size_t length = 10;
  std::srand(std::time(nullptr));
  const int alpha = rand();
  std::vector<int> in(length, alpha);
  std::vector<int> out(length);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(in.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  kovalev_k_bubble_sort_oddeven_transposition_seq::BubbleSortOddEvenTransposition<int> tmpTaskSeq(taskSeq);
  ASSERT_TRUE(tmpTaskSeq.validation());
  tmpTaskSeq.pre_processing();
  tmpTaskSeq.run();
  tmpTaskSeq.post_processing();
  int count_viol = 0;
  for (size_t i = 0; i < length; i++) {
    if (out[i] != in[i]) count_viol++;
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_bubble_sort_oddeven_transposition_seq, Test_1000_int) {
  const size_t length = 1000;
  std::vector<int> in(length);
  std::srand(std::time(nullptr));
  for (size_t i = 0; i < length; i++) in[i] = rand() * std::pow(-1, rand());
  std::vector<int> out(length);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(in.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  kovalev_k_bubble_sort_oddeven_transposition_seq::BubbleSortOddEvenTransposition<int> tmpTaskSeq(taskSeq);
  ASSERT_TRUE(tmpTaskSeq.validation());
  tmpTaskSeq.pre_processing();
  tmpTaskSeq.run();
  tmpTaskSeq.post_processing();
  std::sort(in.begin(), in.end(), [](int a, int b) { return a < b; });
  int count_viol = 0;
  for (size_t i = 0; i < length; i++) {
    if (out[i] != in[i]) count_viol++;
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_bubble_sort_oddeven_transposition_seq, Test_10000_int) {
  const size_t length = 10000;
  std::vector<int> in(length);
  std::srand(std::time(nullptr));
  for (size_t i = 0; i < length; i++) in[i] = rand() * std::pow(-1, rand());
  std::vector<int> out(length);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(in.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  kovalev_k_bubble_sort_oddeven_transposition_seq::BubbleSortOddEvenTransposition<int> tmpTaskSeq(taskSeq);
  ASSERT_TRUE(tmpTaskSeq.validation());
  tmpTaskSeq.pre_processing();
  tmpTaskSeq.run();
  tmpTaskSeq.post_processing();
  std::sort(in.begin(), in.end(), [](int a, int b) { return a < b; });
  int count_viol = 0;
  for (size_t i = 0; i < length; i++) {
    if (out[i] != in[i]) count_viol++;
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_bubble_sort_oddeven_transposition_seq, Test_No_viol_10_double) {
  const size_t length = 10;
  std::srand(std::time(nullptr));
  const double alpha = rand();
  std::vector<double> in(length, alpha);
  std::vector<double> out(length);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(in.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  kovalev_k_bubble_sort_oddeven_transposition_seq::BubbleSortOddEvenTransposition<double> tmpTaskSeq(taskSeq);
  ASSERT_TRUE(tmpTaskSeq.validation());
  tmpTaskSeq.pre_processing();
  tmpTaskSeq.run();
  tmpTaskSeq.post_processing();
  int count_viol = 0;
  for (size_t i = 0; i < length; i++) {
    if (out[i] != in[i]) count_viol++;
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_bubble_sort_oddeven_transposition_seq, Test_1000_double) {
  const size_t length = 1000;
  std::vector<double> in(length);
  auto max = static_cast<double>(1000000);
  auto min = static_cast<double>(-1000000);
  std::srand(std::time(nullptr));
  for (size_t i = 0; i < length; i++) in[i] = min + static_cast<double>(rand()) / RAND_MAX * (max - min);
  std::vector<double> out(length);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(in.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  kovalev_k_bubble_sort_oddeven_transposition_seq::BubbleSortOddEvenTransposition<double> tmpTaskSeq(taskSeq);
  ASSERT_TRUE(tmpTaskSeq.validation());
  tmpTaskSeq.pre_processing();
  tmpTaskSeq.run();
  tmpTaskSeq.post_processing();
  std::sort(in.begin(), in.end(), [](double a, double b) { return a < b; });
  int count_viol = 0;
  for (size_t i = 0; i < length; i++) {
    if (out[i] != in[i]) count_viol++;
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_bubble_sort_oddeven_transposition_seq, Test_10000_double) {
  const size_t length = 10000;
  std::vector<double> in(length);
  auto max = static_cast<double>(1000000);
  auto min = static_cast<double>(-1000000);
  std::srand(std::time(nullptr));
  for (size_t i = 0; i < length; i++) in[i] = min + static_cast<double>(rand()) / RAND_MAX * (max - min);
  std::vector<double> out(length);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(in.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  kovalev_k_bubble_sort_oddeven_transposition_seq::BubbleSortOddEvenTransposition<double> tmpTaskSeq(taskSeq);
  ASSERT_TRUE(tmpTaskSeq.validation());
  tmpTaskSeq.pre_processing();
  tmpTaskSeq.run();
  tmpTaskSeq.post_processing();
  std::sort(in.begin(), in.end(), [](double a, double b) { return a < b; });
  int count_viol = 0;
  for (size_t i = 0; i < length; i++) {
    if (out[i] != in[i]) count_viol++;
  }
  ASSERT_EQ(count_viol, 0);
}