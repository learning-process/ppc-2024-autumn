// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include "mpi/kovalev_k_num_of_orderly_violations/include/header.hpp"

TEST(kovalev_k_num_of_orderly_violations_mpi, Test_NoOV_viol_0_int_) {
  const size_t length = 100;
  const int alpha = 1;
  // Create data
  std::vector<int> in(length, alpha);
  std::vector<size_t> out(1, 0);
  boost::mpi::communicator world;
  // Create TaskData
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
  const int alpha = 1;
  // Create data
  std::vector<int> in(length, alpha);
  std::vector<size_t> out(1, 0);
  for (size_t i = 0; i < length; i++)
  {
    in[i] = 2 * length - i;
  }
  boost::mpi::communicator world;
  // Create TaskData
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
  size_t result = length-1;
  if (world.rank() == 0) {
    ASSERT_EQ(result, out[0]);
  }
}

TEST(kovalev_k_num_of_orderly_violations_mpi, Test_NoOV_len_10_int_) {
  const size_t length = 10;
  const int alpha = 1;
  // Create data
  std::vector<int> in(length, alpha);
  std::vector<size_t> out(1, 0);
  in[1] = -1;
  boost::mpi::communicator world;
  // Create TaskData
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
  size_t result = 1;
  if (world.rank() == 0) {
    ASSERT_EQ(result, out[0]);
  }
}

TEST(kovalev_k_num_of_orderly_violations_mpi, Test_NoOV_len_10000_int_) {
  const size_t length = 10000;
  // Create data
  std::vector<int> in(length);
  for (size_t i = 0; i < length; i++) in[i] = i * 2;
  in[0] = 500;
  in[2] *= 100;
  in[8] *= 3;
  in[21] *= 15;
  in[48] -= 10;
  in[654] += 7;
  in[885] /= 5;
  in[7888] += 48;
  in[71] *= 965;
  in[666] = 532;
  in[228] = 666;
  std::vector<size_t> out(1, 0);
  boost::mpi::communicator world;
  // Create TaskData
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
  size_t result = 11;
  if (world.rank() == 0) {
    ASSERT_EQ(result, out[0]);
  }
}

TEST(kovalev_k_num_of_orderly_violations_mpi, Test_NoOV_viol_0_double_) {
  const size_t length = 100;
  const double alpha = 154.665;
  // Create data
  std::vector<double> in(length, alpha);
  std::vector<size_t> out(1, 0);
  boost::mpi::communicator world;
  // Create TaskData
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
  // Create data
  std::vector<double> in(length);
  std::vector<size_t> out(1, 0);
  for (size_t i = 0; i < length; i++) {
    in[i] = 2 * length - i;
  }
  boost::mpi::communicator world;
  // Create TaskData
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
  size_t result = length-1;
  if (world.rank() == 0) {
    ASSERT_EQ(result, out[0]);
  }
}

TEST(kovalev_k_num_of_orderly_violations_mpi, Test_NoOV_len_10_double_) {
  const size_t length = 10;
  const double alpha = 1.78897;
  // Create data
  std::vector<double> in(length, alpha);
  std::vector<size_t> out(1, 0);
  in[1] = -1;
  boost::mpi::communicator world;
  // Create TaskData
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
  size_t result = 1;
  if (world.rank() == 0) {
    ASSERT_EQ(result, out[0]);
  }
}

TEST(kovalev_k_num_of_orderly_violations_mpi, Test_NoOV_len_1000_double) {
  const size_t length = 10000;
  const double alpha = 70.782;
  // Create data
  std::vector<double> in(length);
  for (size_t i = 0; i < length; i++) in[i] = i * 2;
  in[0] = 500 - alpha;
  in[2] *= -10.756;
  in[8] *= 37.07898;
  in[21] *= 15.0245;
  in[48] -= 10 * alpha;
  in[654] += 7.00;
  in[885] /= 50044.25;
  in[7888] += 48.4;
  in[71] *= 965.7634;
  in[666] = 532.8976;
  in[228] = 666.00001;
  std::vector<size_t> out(1, 0);
  boost::mpi::communicator world;
  // Create TaskData
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
  size_t result = 11;
  if (world.rank() == 0) {
    ASSERT_EQ(result, out[0]);
  }
}