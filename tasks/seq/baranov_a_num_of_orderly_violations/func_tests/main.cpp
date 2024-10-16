#include <gtest/gtest.h>

#include "seq/baranov_a_num_of_orderly_violations/include/header.hpp"
TEST(baranov_a_num_of_orderly_violations_seq, Test_viol_10_int) {
  const int N = 10;
  // Create data
  std::vector<int> arr(N);
  std::random_device rd;
  std::default_random_engine reng(rd());
  std::uniform_int_distribution<int> dist(0, N);
  std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  data_seq->inputs_count.emplace_back(arr.size());
  std::vector<int> out(1);
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(1);
  baranov_a_num_of_orderly_violations_seq::num_of_orderly_violations<int, int> test1(data_seq);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  int num = test1.seq_proc(arr);
  ASSERT_EQ(num, out[0]);
}
TEST(baranov_a_num_of_orderly_violations_seq, Test_viol_100_int) {
  const int N = 100;
  // Create data
  std::vector<int> arr(N);
  std::random_device rd;
  std::default_random_engine reng(rd());
  std::uniform_int_distribution<int> dist(0, N);
  std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  data_seq->inputs_count.emplace_back(arr.size());
  std::vector<int> out(1);
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(1);
  baranov_a_num_of_orderly_violations_seq::num_of_orderly_violations<int, int> test1(data_seq);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  int num = test1.seq_proc(arr);
  ASSERT_EQ(num, out[0]);
}
TEST(baranov_a_num_of_orderly_violations_seq, Test_viol_100_double) {
  const int N = 100;
  // Create data
  std::vector<int> arr(N);
  std::random_device rd;
  std::default_random_engine reng(rd());
  std::uniform_int_distribution<int> dist(0, N);
  std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  data_seq->inputs_count.emplace_back(arr.size());
  std::vector<int> out(1);
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(1);
  baranov_a_num_of_orderly_violations_seq::num_of_orderly_violations<int, int> test1(data_seq);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  int num = test1.seq_proc(arr);
  ASSERT_EQ(num, out[0]);
}

TEST(baranov_a_num_of_orderly_violations_seq, Test_viol_0_double) {
  const int N = 0;
  // Create data
  std::vector<int> arr(N);
  std::random_device rd;
  std::default_random_engine reng(rd());
  std::uniform_int_distribution<int> dist(0, N);
  std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  data_seq->inputs_count.emplace_back(arr.size());
  std::vector<int> out(1);
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(1);
  baranov_a_num_of_orderly_violations_seq::num_of_orderly_violations<int, int> test1(data_seq);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  int num = test1.seq_proc(arr);
  ASSERT_EQ(num, out[0]);
}
TEST(baranov_a_num_of_orderly_violations_seq, Test_viol_1000_double) {
  const int N = 1000;
  // Create data
  std::vector<int> arr(N);
  std::random_device rd;
  std::default_random_engine reng(rd());
  std::uniform_int_distribution<int> dist(0, N);
  std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  data_seq->inputs_count.emplace_back(arr.size());
  std::vector<int> out(1);
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(1);
  baranov_a_num_of_orderly_violations_seq::num_of_orderly_violations<int, int> test1(data_seq);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  int num = test1.seq_proc(arr);
  ASSERT_EQ(num, out[0]);
}
TEST(baranov_a_num_of_orderly_violations_seq, Test_viol_10000_double) {
  const int N = 10000;
  // Create data
  std::vector<int> arr(N);
  std::random_device rd;
  std::default_random_engine reng(rd());
  std::uniform_int_distribution<int> dist(0, N);
  std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  data_seq->inputs_count.emplace_back(arr.size());
  std::vector<int> out(1);
  data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  data_seq->outputs_count.emplace_back(1);
  baranov_a_num_of_orderly_violations_seq::num_of_orderly_violations<int, int> test1(data_seq);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  int num = test1.seq_proc(arr);
  ASSERT_EQ(num, out[0]);
}
int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
