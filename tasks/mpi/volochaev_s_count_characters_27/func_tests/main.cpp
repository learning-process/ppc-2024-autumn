#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/serialization/map.hpp>
#include <vector>

#include "mpi/volochaev_s_count_characters_27/include/ops_mpi.hpp"

TEST(volochaev_s_count_characters_27_MPI, Test_0) {
  boost::mpi::communicator world;
  std::vector<std::string> global_vec;
  std::vector<int32_t> global_diff(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int size_str1 = 240;
    const int size_str2 = 120;
    global_vec = {volochaev_s_count_characters_27_mpi::get_random_string(size_str1)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_diff.data()));
    taskDataPar->outputs_count.emplace_back(global_diff.size());
  }

  volochaev_s_count_characters_27_mpi::Lab1_27_mpi testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_diff(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_diff.data()));
    taskDataSeq->outputs_count.emplace_back(reference_diff.size());

    // Create Task
    volochaev_s_count_characters_27_mpi::Lab1_27_seq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), false);

    ASSERT_EQ(reference_diff[0], global_diff[0]);
  }
}

TEST(volochaev_s_count_characters_27_MPI, Test_1) {
  boost::mpi::communicator world;
  std::vector<std::string> global_vec;
  std::vector<int32_t> global_diff(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int size_str1 = 240;
    const int size_str2 = 120;
    global_vec = {volochaev_s_count_characters_27_mpi::get_random_string(size_str1),
                  volochaev_s_count_characters_27_mpi::get_random_string(size_str2)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_diff.data()));
    taskDataPar->outputs_count.emplace_back(global_diff.size());
  }

  volochaev_s_count_characters_27_mpi::Lab1_27_mpi testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_diff(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_diff.data()));
    taskDataSeq->outputs_count.emplace_back(reference_diff.size());

    // Create Task
    volochaev_s_count_characters_27_mpi::Lab1_27_seq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_diff[0], global_diff[0]);
  }
}

TEST(volochaev_s_count_characters_27_MPI, Test_2) {
  boost::mpi::communicator world;
  std::vector<std::string> global_vec;
  std::vector<int32_t> global_diff(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int string_sz = 120;
    std::string s = volochaev_s_count_characters_27_mpi::get_random_string(string_sz);
    global_vec = {s, s};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_diff.data()));
    taskDataPar->outputs_count.emplace_back(global_diff.size());
  }

  volochaev_s_count_characters_27_mpi::Lab1_27_mpi testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_diff(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_diff.data()));
    taskDataSeq->outputs_count.emplace_back(reference_diff.size());

    // Create Task
    volochaev_s_count_characters_27_mpi::Lab1_27_seq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_diff[0], global_diff[0]);
  }
}

TEST(volochaev_s_count_characters_27_MPI, Test_3) {
  boost::mpi::communicator world;
  std::vector<std::string> global_vec;
  std::vector<int32_t> global_max(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int size_str1 = 240;
    const int size_str2 = 120;
    global_vec = {volochaev_s_count_characters_27_mpi::get_random_string(size_str2),
                  volochaev_s_count_characters_27_mpi::get_random_string(size_str1)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  volochaev_s_count_characters_27_mpi::Lab1_27_mpi testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    volochaev_s_count_characters_27_mpi::Lab1_27_seq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(volochaev_s_count_characters_27_MPI, Test_4) {
  boost::mpi::communicator world;
  std::vector<std::string> global_vec;
  std::vector<int32_t> global_max(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int size_str = 120;
    global_vec = {volochaev_s_count_characters_27_mpi::get_random_string(size_str),
                  volochaev_s_count_characters_27_mpi::get_random_string(size_str)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  volochaev_s_count_characters_27_mpi::Lab1_27_mpi testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_max(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    volochaev_s_count_characters_27_mpi::Lab1_27_seq testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}