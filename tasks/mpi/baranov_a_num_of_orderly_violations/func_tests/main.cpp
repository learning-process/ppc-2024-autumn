#include <gtest/gtest.h>

#include "mpi/baranov_a_num_of_orderly_violations/include/header.hpp"

TEST(baranov_a_num_of_orderly_violations_mpi, Test_viol_0_int) {
  const int N = 0;
  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<int> out(1);
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::random_device rd;
    std::default_random_engine reng(rd());
    std::uniform_int_distribution<int> dist(0, arr.size());
    std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
    data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data_seq->inputs_count.emplace_back(arr.size());
    data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data_seq->outputs_count.emplace_back(1);
  }
  baranov_a_num_of_orderly_violations_mpi::num_of_orderly_violations<int, int> test1(data_seq);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  int num = test1.seq_proc(arr);
  ASSERT_EQ(out[0], num);
}
TEST(baranov_a_num_of_orderly_violations_mpi, Test_viol_100_int) {
  const int N = 100;
  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<int> out(1);
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::random_device rd;
    std::default_random_engine reng(rd());
    std::uniform_int_distribution<int> dist(0, arr.size());
    std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
    data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data_seq->inputs_count.emplace_back(arr.size());
    data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data_seq->outputs_count.emplace_back(1);
  }
  baranov_a_num_of_orderly_violations_mpi::num_of_orderly_violations<int, int> test1(data_seq);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  int num = test1.seq_proc(arr);
  ASSERT_EQ(out[0], num);
}

TEST(baranov_a_num_of_orderly_violations_mpi, Test_viol_10000_int) {
  const int N = 10000;
  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<int> out(1);
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::random_device rd;
    std::default_random_engine reng(rd());
    std::uniform_int_distribution<int> dist(0, arr.size());
    std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
    data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data_seq->inputs_count.emplace_back(arr.size());
    data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data_seq->outputs_count.emplace_back(1);
  }
  baranov_a_num_of_orderly_violations_mpi::num_of_orderly_violations<int, int> test1(data_seq);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  int num = test1.seq_proc(arr);
  ASSERT_EQ(num, out[0]);
}
TEST(baranov_a_num_of_orderly_violations_mpi, Test_viol_0_double) {
  const int N = 0;
  // Create data
  boost::mpi::communicator world;
  std::vector<double> arr(N);
  std::vector<int> out(1);
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::random_device rd;
    std::default_random_engine reng(rd());
    std::uniform_int_distribution<int> dist(0, arr.size());
    std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
    data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data_seq->inputs_count.emplace_back(arr.size());
    data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data_seq->outputs_count.emplace_back(1);
  }
  baranov_a_num_of_orderly_violations_mpi::num_of_orderly_violations<double, int> test1(data_seq);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  int num = test1.seq_proc(arr);
  ASSERT_EQ(num, out[0]);
}

TEST(baranov_a_num_of_orderly_violations_mpi, Test_viol_10000_double) {
  const int N = 10000;
  // Create data
  boost::mpi::communicator world;
  std::vector<double> arr(N);
  std::vector<int> out(1);
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::random_device rd;
    std::default_random_engine reng(rd());
    std::uniform_int_distribution<int> dist(0, arr.size());
    std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
    data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data_seq->inputs_count.emplace_back(arr.size());
    data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data_seq->outputs_count.emplace_back(1);
  }
  baranov_a_num_of_orderly_violations_mpi::num_of_orderly_violations<double, int> test1(data_seq);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  int num = test1.seq_proc(arr);
  ASSERT_EQ(num, out[0]);
}
TEST(baranov_a_num_of_orderly_violations_mpi, Test_viol_size_of_vec_is_equal_to_world_size) {
  // Create data
  boost::mpi::communicator world;
  const int N = world.size();
  std::vector<double> arr(N);
  std::vector<int> out(1);
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::random_device rd;
    std::default_random_engine reng(rd());
    std::uniform_int_distribution<int> dist(0, arr.size());
    std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
    data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data_seq->inputs_count.emplace_back(arr.size());
    data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data_seq->outputs_count.emplace_back(1);
  }
  baranov_a_num_of_orderly_violations_mpi::num_of_orderly_violations<double, int> test1(data_seq);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  int num = test1.seq_proc(arr);
  ASSERT_EQ(num, out[0]);
}
TEST(baranov_a_num_of_orderly_violations_mpi, Test_viol_size_of_vec_is_less_than_world_size) {
  // Create data
  boost::mpi::communicator world;
  const int N = world.size() - 1;
  std::vector<double> arr(N);
  std::vector<int> out(1);
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::random_device rd;
    std::default_random_engine reng(rd());
    std::uniform_int_distribution<int> dist(0, arr.size());
    std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
    data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data_seq->inputs_count.emplace_back(arr.size());
    data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data_seq->outputs_count.emplace_back(1);
  }
  baranov_a_num_of_orderly_violations_mpi::num_of_orderly_violations<double, int> test1(data_seq);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  int num = test1.seq_proc(arr);
  ASSERT_EQ(num, out[0]);
}

TEST(baranov_a_num_of_orderly_violations_mpi, Test_viol_100000_unsigned_int) {
  const int N = 100000;
  // Create data
  boost::mpi::communicator world;
  std::vector<unsigned> arr(N);
  std::vector<int> out(1);
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::random_device rd;
    std::default_random_engine reng(rd());
    std::uniform_int_distribution<int> dist(0, arr.size());
    std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
    data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data_seq->inputs_count.emplace_back(arr.size());
    data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data_seq->outputs_count.emplace_back(1);
  }
  baranov_a_num_of_orderly_violations_mpi::num_of_orderly_violations<unsigned, int> test1(data_seq);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  int num = test1.seq_proc(arr);
  ASSERT_EQ(num, out[0]);
}

TEST(baranov_a_num_of_orderly_violations_mpi, Test_viol_odd_numbers_int_1) {
  const int N = 78527;
  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<int> out(1);
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::random_device rd;
    std::default_random_engine reng(rd());
    std::uniform_int_distribution<int> dist(0, arr.size());
    std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
    data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data_seq->inputs_count.emplace_back(arr.size());
    data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data_seq->outputs_count.emplace_back(1);
  }
  baranov_a_num_of_orderly_violations_mpi::num_of_orderly_violations<int, int> test1(data_seq);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  int num = test1.seq_proc(arr);
  ASSERT_EQ(out[0], num);
}
TEST(baranov_a_num_of_orderly_violations_mpi, Test_viol_odd_numbers_int_2) {
  const int N = 2356895;
  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<int> out(1);
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::random_device rd;
    std::default_random_engine reng(rd());
    std::uniform_int_distribution<int> dist(0, arr.size());
    std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
    data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data_seq->inputs_count.emplace_back(arr.size());
    data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data_seq->outputs_count.emplace_back(1);
  }
  baranov_a_num_of_orderly_violations_mpi::num_of_orderly_violations<int, int> test1(data_seq);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  int num = test1.seq_proc(arr);
  ASSERT_EQ(out[0], num);
}
TEST(baranov_a_num_of_orderly_violations_mpi, Test_viol_odd_numbers_int_3) {
  const int N = 17;
  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<int> out(1);
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::random_device rd;
    std::default_random_engine reng(rd());
    std::uniform_int_distribution<int> dist(0, arr.size());
    std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
    data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data_seq->inputs_count.emplace_back(arr.size());
    data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data_seq->outputs_count.emplace_back(1);
  }
  baranov_a_num_of_orderly_violations_mpi::num_of_orderly_violations<int, int> test1(data_seq);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  int num = test1.seq_proc(arr);
  ASSERT_EQ(out[0], num);
}
