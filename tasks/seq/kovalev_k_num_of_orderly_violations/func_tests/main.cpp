#include <gtest/gtest.h>

#include "seq/kovalev_k_num_of_orderly_violations/include/header.hpp"

TEST(kovalev_k_num_of_orderly_violations_seq, zero_length) {
  std::vector<int> in;
  std::vector<size_t> out;
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(in.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  kovalev_k_num_of_orderly_violations_seq::NumOfOrderlyViolations<int> tmpTaskSeq(taskSeq);
  ASSERT_FALSE(tmpTaskSeq.validation());
}

TEST(kovalev_k_num_of_orderly_violations_seq, Test_NoOV_viol_0_int_) {
  const size_t length = 10;
  const int alpha = 1;
  std::vector<int> in(length, alpha);
  std::vector<size_t> out(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(in.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  kovalev_k_num_of_orderly_violations_seq::NumOfOrderlyViolations<int> tmpTaskSeq(taskSeq);
  ASSERT_EQ(tmpTaskSeq.validation(), true);
  tmpTaskSeq.pre_processing();
  tmpTaskSeq.run();
  tmpTaskSeq.post_processing();
  size_t result = 0;
  ASSERT_EQ(result, out[0]);
}

TEST(kovalev_k_num_of_orderly_violations_seq, Test_NoOV_len_10_rand_int_) {
  const size_t length = 10;
  std::vector<int> in(length);
  std::vector<size_t> out(1, 0);
  std::srand(std::time(nullptr));
  for (size_t i = 0; i < length; i++) in[i] = rand() * std::pow(-1, rand());
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(in.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  kovalev_k_num_of_orderly_violations_seq::NumOfOrderlyViolations<int> tmpTaskSeq(taskSeq);
  ASSERT_EQ(tmpTaskSeq.validation(), true);
  tmpTaskSeq.pre_processing();
  tmpTaskSeq.run();
  tmpTaskSeq.post_processing();
  size_t result = 0;
  for (size_t i = 1; i < length; i++)
    if (in[i - 1] > in[i]) result++;
  ASSERT_EQ(result, out[0]);
}

TEST(kovalev_k_num_of_orderly_violations_seq, Test_NoOV_len_10000_rand_int_) {
  const size_t length = 10000;
  std::vector<int> in(length);
  std::vector<size_t> out(1, 0);
  std::srand(std::time(nullptr));
  for (size_t i = 0; i < length; i++) in[i] = rand() * std::pow(-1, rand());
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(in.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  kovalev_k_num_of_orderly_violations_seq::NumOfOrderlyViolations<int> tmpTaskSeq(taskSeq);
  ASSERT_EQ(tmpTaskSeq.validation(), true);
  tmpTaskSeq.pre_processing();
  tmpTaskSeq.run();
  tmpTaskSeq.post_processing();
  size_t result = 0;
  for (size_t i = 1; i < length; i++)
    if (in[i - 1] > in[i]) result++;
  ASSERT_EQ(result, out[0]);
}

TEST(kovalev_k_num_of_orderly_violations_seq, Test_NoOV_viol_0_double_) {
  const size_t length = 10;
  auto max = static_cast<double>(1000000);
  auto min = static_cast<double>(-1000000);
  std::srand(std::time(nullptr));
  const double alpha = min + static_cast<double>(rand()) / RAND_MAX * (max - min);
  std::vector<double> in(length, alpha);
  std::vector<size_t> out(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(in.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  kovalev_k_num_of_orderly_violations_seq::NumOfOrderlyViolations<double> tmpTaskSeq(taskSeq);
  ASSERT_EQ(tmpTaskSeq.validation(), true);
  tmpTaskSeq.pre_processing();
  tmpTaskSeq.run();
  tmpTaskSeq.post_processing();
  size_t result = 0;
  ASSERT_EQ(result, out[0]);
}

TEST(kovalev_k_num_of_orderly_violations_seq, Test_NoOV_len_10_rand_double_) {
  const size_t length = 10;
  std::vector<double> in(length);
  auto max = static_cast<double>(1000000);
  auto min = static_cast<double>(-1000000);
  std::srand(std::time(nullptr));
  for (size_t i = 0; i < length; i++) in[i] = min + static_cast<double>(rand()) / RAND_MAX * (max - min);
  std::vector<size_t> out(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(in.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  kovalev_k_num_of_orderly_violations_seq::NumOfOrderlyViolations<double> tmpTaskSeq(taskSeq);
  ASSERT_EQ(tmpTaskSeq.validation(), true);
  tmpTaskSeq.pre_processing();
  tmpTaskSeq.run();
  tmpTaskSeq.post_processing();
  size_t result = 0;
  for (size_t i = 1; i < length; i++)
    if (in[i - 1] > in[i]) result++;
  ASSERT_EQ(result, out[0]);
}

TEST(kovalev_k_num_of_orderly_violations_seq, Test_NoOV_len_10000_rand_double) {
  const size_t length = 10000;
  std::vector<double> in(length);
  auto max = static_cast<double>(1000000);
  auto min = static_cast<double>(-1000000);
  std::srand(std::time(nullptr));
  for (size_t i = 0; i < length; i++) in[i] = min + static_cast<double>(rand()) / RAND_MAX * (max - min);
  std::vector<size_t> out(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(in.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  kovalev_k_num_of_orderly_violations_seq::NumOfOrderlyViolations<double> tmpTaskSeq(taskSeq);
  ASSERT_EQ(tmpTaskSeq.validation(), true);
  tmpTaskSeq.pre_processing();
  tmpTaskSeq.run();
  tmpTaskSeq.post_processing();
  size_t result = 0;
  for (size_t i = 1; i < length; i++)
    if (in[i - 1] > in[i]) result++;
  ASSERT_EQ(result, out[0]);
}

TEST(kovalev_k_num_of_orderly_violations_seq, Test_NoOV_len_1000000_rand_double) {
  const size_t length = 1000000;
  std::vector<double> in(length);
  auto max = static_cast<double>(1000000);
  auto min = static_cast<double>(-1000000);
  std::srand(std::time(nullptr));
  for (size_t i = 0; i < length; i++) in[i] = min + static_cast<double>(rand()) / RAND_MAX * (max - min);
  std::vector<size_t> out(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(in.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  kovalev_k_num_of_orderly_violations_seq::NumOfOrderlyViolations<double> tmpTaskSeq(taskSeq);
  ASSERT_EQ(tmpTaskSeq.validation(), true);
  tmpTaskSeq.pre_processing();
  tmpTaskSeq.run();
  tmpTaskSeq.post_processing();
  size_t result = 0;
  for (size_t i = 1; i < length; i++)
    if (in[i - 1] > in[i]) result++;
  ASSERT_EQ(result, out[0]);
}