// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/frolova_e_num_of_letters/include/ops_mpi.hpp"
/*
TEST(frolova_e_num_of_letters_mpi, Test_100_symbols) {
  boost::mpi::communicator world;
  std::string global_str;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_ = 100;
    global_str = frolova_e_num_of_letters_mpi::GenStr(count_size_);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataPar->inputs_count.emplace_back(global_str.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  frolova_e_num_of_letters_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_sum(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    // Create Task
    frolova_e_num_of_letters_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}

TEST(frolova_e_num_of_letters_mpi, Test_with_number) {
  boost::mpi::communicator world;
  std::string global_str;
  std::vector<int32_t> global_diff(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_ = 240;
    global_str = frolova_e_num_of_letters_mpi::GenStr(count_size_);
    global_str.push_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataPar->inputs_count.emplace_back(global_str.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_diff.data()));
    taskDataPar->outputs_count.emplace_back(global_diff.size());
  }

  frolova_e_num_of_letters_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_diff(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_diff.data()));
    taskDataSeq->outputs_count.emplace_back(reference_diff.size());

    // Create Task
    frolova_e_num_of_letters_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_diff[0], global_diff[0]);
  }
}

TEST(frolova_e_num_of_letters_mpi, Test_only_numbers) {
  boost::mpi::communicator world;
  std::string global_str;
  std::vector<int32_t> global_diff(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_str = "1234567890";
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataPar->inputs_count.emplace_back(global_str.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_diff.data()));
    taskDataPar->outputs_count.emplace_back(global_diff.size());
  }

  frolova_e_num_of_letters_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_diff(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_diff.data()));
    taskDataSeq->outputs_count.emplace_back(reference_diff.size());

    // Create Task
    frolova_e_num_of_letters_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_diff[0], global_diff[0]);
  }
}

TEST(frolova_e_num_of_letters_mpi, Test_empty_str) {
   boost::mpi::communicator world;
   std::string global_str;
   std::vector<int32_t> global_diff(1, 0);
    // Create TaskData
   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

   if (world.rank() == 0) {
     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
     taskDataPar->inputs_count.emplace_back(global_str.size());
     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_diff.data()));
     taskDataPar->outputs_count.emplace_back(global_diff.size());
   }

   frolova_e_num_of_letters_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
   ASSERT_EQ(testMpiTaskParallel.validation(), false);
}

TEST(frolova_e_num_of_letters_mpi, Test_different_symbols) {
   boost::mpi::communicator world;
   std::string global_str;
   std::vector<int32_t> global_diff(1, 0);
    // Create TaskData
   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

   if (world.rank() == 0) {
     global_str = "1234567890;;-a";
     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
     taskDataPar->inputs_count.emplace_back(global_str.size());
     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_diff.data()));
     taskDataPar->outputs_count.emplace_back(global_diff.size());
   }

   frolova_e_num_of_letters_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
   ASSERT_EQ(testMpiTaskParallel.validation(), true);
   testMpiTaskParallel.pre_processing();
   testMpiTaskParallel.run();
   testMpiTaskParallel.post_processing();

   if (world.rank() == 0) {
      // Create data
     std::vector<int32_t> reference_diff(1, 0);

      // Create TaskData
     std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
     taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
     taskDataSeq->inputs_count.emplace_back(global_str.size());
     taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_diff.data()));
     taskDataSeq->outputs_count.emplace_back(reference_diff.size());

      // Create Task
     frolova_e_num_of_letters_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
     ASSERT_EQ(testMpiTaskSequential.validation(), true);
     testMpiTaskSequential.pre_processing();
     testMpiTaskSequential.run();
     testMpiTaskSequential.post_processing();

     ASSERT_EQ(reference_diff[0], global_diff[0]);
   }
}
*/