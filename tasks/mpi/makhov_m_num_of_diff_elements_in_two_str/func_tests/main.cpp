// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/makhov_m_num_of_diff_elements_in_two_str/include/ops_mpi.hpp"

TEST(makhov_m_num_of_diff_elements_in_two_str_mpi, SameSizeRandomStrings) {
  boost::mpi::communicator world;
  std::string str1;
  std::string str2;
  std::vector<int32_t> global_sum(1, 0);
  std::vector<int32_t> reference_sum(1, 0);
  std::random_device dev;
  std::mt19937 gen(dev());
  size_t size = 10;
  char min = '0';
  char max = '9';
  for (size_t i = 0; i < size; i++) {
    str1 += (char)(min + gen() % (max - min + 1));
    str2 += (char)(min + gen() % (max - min + 1));
  }
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str1.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str2.data()));
    taskDataPar->inputs_count.emplace_back(str1.size());
    taskDataPar->inputs_count.emplace_back(str2.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  // Create Task
  makhov_m_num_of_diff_elements_in_two_str_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str1.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str2.data()));
    taskDataSeq->inputs_count.emplace_back(str1.size());
    taskDataSeq->inputs_count.emplace_back(str2.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    // Create Task
    makhov_m_num_of_diff_elements_in_two_str_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}

TEST(makhov_m_num_of_diff_elements_in_two_str_mpi, DiffSizeRandomStrings) {
  boost::mpi::communicator world;
  std::string str1;
  std::string str2;
  std::vector<int32_t> global_sum(1, 0);
  std::vector<int32_t> reference_sum(1, 0);
  std::random_device dev;
  std::mt19937 gen(dev());
  size_t size1 = 10;
  size_t size2 = 15;
  char min = '0';
  char max = '9';
  for (size_t i = 0; i < size1; i++) {
    str1 += (char)(min + gen() % (max - min + 1));
  }
  for (size_t i = 0; i < size2; i++) {
    str2 += (char)(min + gen() % (max - min + 1));
  }
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str1.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str2.data()));
    taskDataPar->inputs_count.emplace_back(str1.size());
    taskDataPar->inputs_count.emplace_back(str2.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  // Create Task
  makhov_m_num_of_diff_elements_in_two_str_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str1.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str2.data()));
    taskDataSeq->inputs_count.emplace_back(str1.size());
    taskDataSeq->inputs_count.emplace_back(str2.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    // Create Task
    makhov_m_num_of_diff_elements_in_two_str_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}

TEST(makhov_m_num_of_diff_elements_in_two_str_mpi, SameEmptySymbolsStrings) {
  boost::mpi::communicator world;
  std::string str1;
  std::string str2;
  std::vector<int32_t> global_sum(1, 0);
  std::vector<int32_t> reference_sum(1, 0);
  str1 = "   ";
  str2 = str1;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str1.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str2.data()));
    taskDataPar->inputs_count.emplace_back(str1.size());
    taskDataPar->inputs_count.emplace_back(str2.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  // Create Task
  makhov_m_num_of_diff_elements_in_two_str_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str1.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str2.data()));
    taskDataSeq->inputs_count.emplace_back(str1.size());
    taskDataSeq->inputs_count.emplace_back(str2.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    // Create Task
    makhov_m_num_of_diff_elements_in_two_str_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}

TEST(makhov_m_num_of_diff_elements_in_two_str_mpi, DiffSizeEmptySymbolsStrings) {
  boost::mpi::communicator world;
  std::string str1;
  std::string str2;
  std::vector<int32_t> global_sum(1, 0);
  std::vector<int32_t> reference_sum(1, 0);
  str1 = "   ";
  str2 = "      ";
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str1.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str2.data()));
    taskDataPar->inputs_count.emplace_back(str1.size());
    taskDataPar->inputs_count.emplace_back(str2.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  // Create Task
  makhov_m_num_of_diff_elements_in_two_str_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str1.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str2.data()));
    taskDataSeq->inputs_count.emplace_back(str1.size());
    taskDataSeq->inputs_count.emplace_back(str2.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    // Create Task
    makhov_m_num_of_diff_elements_in_two_str_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}

TEST(makhov_m_num_of_diff_elements_in_two_str_mpi, EqualSizeDiffStrings) {
  boost::mpi::communicator world;
  std::string str1;
  std::string str2;
  std::vector<int32_t> global_sum(1, 0);
  std::vector<int32_t> reference_sum(1, 0);
  str1 = "Hello, World!!";
  str2 = "Goodbye, World";
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str1.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str2.data()));
    taskDataPar->inputs_count.emplace_back(str1.size());
    taskDataPar->inputs_count.emplace_back(str2.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  // Create Task
  makhov_m_num_of_diff_elements_in_two_str_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str1.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str2.data()));
    taskDataSeq->inputs_count.emplace_back(str1.size());
    taskDataSeq->inputs_count.emplace_back(str2.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    // Create Task
    makhov_m_num_of_diff_elements_in_two_str_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}

TEST(makhov_m_num_of_diff_elements_in_two_str_mpi, DiffSizeDiffStrings) {
  boost::mpi::communicator world;
  std::string str1;
  std::string str2;
  std::vector<int32_t> global_sum(1, 0);
  std::vector<int32_t> reference_sum(1, 0);
  str1 = "1x1xx111";
  str2 = "11111111dfgdfg";
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str1.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str2.data()));
    taskDataPar->inputs_count.emplace_back(str1.size());
    taskDataPar->inputs_count.emplace_back(str2.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  // Create Task
  makhov_m_num_of_diff_elements_in_two_str_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str1.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str2.data()));
    taskDataSeq->inputs_count.emplace_back(str1.size());
    taskDataSeq->inputs_count.emplace_back(str2.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    // Create Task
    makhov_m_num_of_diff_elements_in_two_str_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}

TEST(makhov_m_num_of_diff_elements_in_two_str_mpi, SameStrings) {
  boost::mpi::communicator world;
  std::string str1;
  std::string str2;
  std::vector<int32_t> global_sum(1, 0);
  std::vector<int32_t> reference_sum(1, 0);
  str1 = "Hello, World!";
  str2 = str1;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str1.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str2.data()));
    taskDataPar->inputs_count.emplace_back(str1.size());
    taskDataPar->inputs_count.emplace_back(str2.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  // Create Task
  makhov_m_num_of_diff_elements_in_two_str_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str1.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str2.data()));
    taskDataSeq->inputs_count.emplace_back(str1.size());
    taskDataSeq->inputs_count.emplace_back(str2.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    // Create Task
    makhov_m_num_of_diff_elements_in_two_str_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}

TEST(makhov_m_num_of_diff_elements_in_two_str_mpi, EmptyString) {
  boost::mpi::communicator world;
  std::string str1;
  std::string str2;
  std::vector<int32_t> global_sum(1, 0);
  std::vector<int32_t> reference_sum(1, 0);
  std::random_device dev;
  std::mt19937 gen(dev());
  size_t size = 10;
  char min = '0';
  char max = '9';
  str1 = "";
  for (size_t i = 0; i < size; i++) {
    str2 += (char)(min + gen() % (max - min + 1));
  }
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str1.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str2.data()));
    taskDataPar->inputs_count.emplace_back(str1.size());
    taskDataPar->inputs_count.emplace_back(str2.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  // Create Task
  makhov_m_num_of_diff_elements_in_two_str_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str1.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str2.data()));
    taskDataSeq->inputs_count.emplace_back(str1.size());
    taskDataSeq->inputs_count.emplace_back(str2.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    // Create Task
    makhov_m_num_of_diff_elements_in_two_str_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSequential.validation());
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(10, global_sum[0]);
  }
}
