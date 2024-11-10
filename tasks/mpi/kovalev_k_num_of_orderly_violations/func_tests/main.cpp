#include <gtest/gtest.h>

#include "mpi/kovalev_k_num_of_orderly_violations/include/header.hpp"

TEST(kovalev_k_num_of_orderly_violations_mpi, zero_length) {
  std::vector<int> in;
  std::vector<size_t> out;
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    tmpPar->inputs_count.emplace_back(in.size());
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    tmpPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<int> tmpTaskPar(tmpPar);
  if (world.rank() == 0) {
    ASSERT_FALSE(tmpTaskPar.validation());
  }
}

TEST(kovalev_k_num_of_orderly_violations_mpi, Test_NoOV_viol_0_int_) {
  const size_t length = 100;
  std::srand(std::time(nullptr));
  const int alpha = rand();
  std::vector<int> in(length, alpha);
  std::vector<size_t> out(1, 0);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    tmpPar->inputs_count.emplace_back(in.size());
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    tmpPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<int> tmpTaskPar(tmpPar);
  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();
  size_t result = 0;
  if (world.rank() == 0) {
    ASSERT_EQ(result, out[0]);
  }
}

TEST(kovalev_k_num_of_orderly_violations_mpi, Test_NoOV_len_100_opposite_sort_int_) {
  const size_t length = 100;
  std::srand(std::time(nullptr));
  const int alpha = rand();
  std::vector<int> in(length, alpha);
  std::vector<size_t> out(1, 0);
  for (size_t i = 0; i < length; i++) {
    in[i] = 2 * length - i;
  }
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    tmpPar->inputs_count.emplace_back(in.size());
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    tmpPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<int> tmpTaskPar(tmpPar);
  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();
  size_t result = length - 1;
  if (world.rank() == 0) {
    ASSERT_EQ(result, out[0]);
  }
}

TEST(kovalev_k_num_of_orderly_violations_mpi, Test_NoOV_len_10_rand_int_) {
  const size_t length = 10;
  std::vector<int> in(length);
  std::vector<size_t> out(1, 0);
  std::srand(std::time(nullptr));
  for (size_t i = 0; i < length; i++) in[i] = rand() * std::pow(-1, rand());
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    tmpPar->inputs_count.emplace_back(in.size());
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    tmpPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<int> tmpTaskPar(tmpPar);
  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();
  if (world.rank() == 0) {
    size_t result = 0;
    for (size_t i = 1; i < length; i++)
      if (in[i - 1] > in[i]) result++;
    ASSERT_EQ(result, out[0]);
  }
}

TEST(kovalev_k_num_of_orderly_violations_mpi, Test_NoOV_len_10000_rand_int_) {
  const size_t length = 10000;
  std::vector<int> in(length);
  std::vector<size_t> out(1, 0);
  std::srand(std::time(nullptr));
  for (size_t i = 0; i < length; i++) in[i] = rand() * std::pow(-1, rand());
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    tmpPar->inputs_count.emplace_back(in.size());
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    tmpPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<int> tmpTaskPar(tmpPar);
  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();
  if (world.rank() == 0) {
    size_t result = 0;
    for (size_t i = 1; i < length; i++)
      if (in[i - 1] > in[i]) result++;
    ASSERT_EQ(result, out[0]);
  }
}

TEST(kovalev_k_num_of_orderly_violations_mpi, Test_NoOV_viol_0_double_) {
  const size_t length = 100;
  auto max = static_cast<double>(1000000);
  auto min = static_cast<double>(-1000000);
  std::srand(std::time(nullptr));
  const double alpha = min + static_cast<double>(rand()) / RAND_MAX * (max - min);
  std::vector<double> in(length, alpha);
  std::vector<size_t> out(1, 0);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    tmpPar->inputs_count.emplace_back(in.size());
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    tmpPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<double> tmpTaskPar(tmpPar);
  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();
  size_t result = 0;
  if (world.rank() == 0) {
    ASSERT_EQ(result, out[0]);
  }
}

TEST(kovalev_k_num_of_orderly_violations_mpi, Test_NoOV_len_100_opposite_sort_double_) {
  const size_t length = 100;
  std::srand(std::time(nullptr));
  const double alpha = (static_cast<double>(rand()) - 1) / (RAND_MAX);
  std::vector<double> in(length);
  std::vector<size_t> out(1, 0);
  in[0] = static_cast<double>(length);
  for (size_t i = 1; i < length; i++) {
    in[i] = in[i - 1] * alpha;
  }
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    tmpPar->inputs_count.emplace_back(in.size());
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    tmpPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<double> tmpTaskPar(tmpPar);
  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();
  size_t result = length - 1;
  if (world.rank() == 0) {
    ASSERT_EQ(result, out[0]);
  }
}

TEST(kovalev_k_num_of_orderly_violations_mpi, Test_NoOV_len_10_rand_double_) {
  const size_t length = 10;
  std::vector<double> in(length);
  auto max = static_cast<double>(1000000);
  auto min = static_cast<double>(-1000000);
  std::srand(std::time(nullptr));
  for (size_t i = 0; i < length; i++) in[i] = min + static_cast<double>(rand()) / RAND_MAX * (max - min);
  std::vector<size_t> out(1, 0);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    tmpPar->inputs_count.emplace_back(in.size());
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    tmpPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<double> tmpTaskPar(tmpPar);
  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();
  if (world.rank() == 0) {
    size_t result = 0;
    for (size_t i = 1; i < length; i++)
      if (in[i - 1] > in[i]) result++;
    ASSERT_EQ(result, out[0]);
  }
}

TEST(kovalev_k_num_of_orderly_violations_mpi, Test_NoOV_len_10000_rand_double_) {
  const size_t length = 10000;
  std::vector<double> in(length);
  auto max = static_cast<double>(1000000);
  auto min = static_cast<double>(-1000000);
  std::srand(std::time(nullptr));
  for (size_t i = 0; i < length; i++) in[i] = min + static_cast<double>(rand()) / RAND_MAX * (max - min);
  std::vector<size_t> out(1, 0);
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    tmpPar->inputs_count.emplace_back(in.size());
    tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    tmpPar->outputs_count.emplace_back(out.size());
  }
  kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<double> tmpTaskPar(tmpPar);
  ASSERT_TRUE(tmpTaskPar.validation());
  tmpTaskPar.pre_processing();
  tmpTaskPar.run();
  tmpTaskPar.post_processing();
  if (world.rank() == 0) {
    size_t result = 0;
    for (size_t i = 1; i < length; i++)
      if (in[i - 1] > in[i]) result++;
    ASSERT_EQ(result, out[0]);
  }
}