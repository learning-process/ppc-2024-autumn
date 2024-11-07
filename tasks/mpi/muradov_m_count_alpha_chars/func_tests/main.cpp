#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "mpi/muradov_m_count_alpha_chars/include/ops_mpi.hpp"

TEST(muradov_m_count_alpha_chars_mpi, test_all_alpha_characters) {
  boost::mpi::communicator world;
  std::string global_str;
  std::vector<int32_t> global_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_str = 240;
    global_str = std::string(count_size_str, 'a');
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataPar->inputs_count.emplace_back(global_str.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }

  muradov_m_count_alpha_chars_mpi::AlphaCharCountTaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_count(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_count.data()));
    taskDataSeq->outputs_count.emplace_back(reference_count.size());

    muradov_m_count_alpha_chars_mpi::AlphaCharCountTaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_count[0], global_count[0]);
  }
}

TEST(muradov_m_count_alpha_chars_mpi, test_mixed_characters) {
  boost::mpi::communicator world;
  std::string global_str;
  std::vector<int32_t> global_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_str = 240;
    global_str = std::string(count_size_str, '1');
    for (int i = 0; i < count_size_str; i += 2) {
      global_str[i] = 'b';
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataPar->inputs_count.emplace_back(global_str.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }

  muradov_m_count_alpha_chars_mpi::AlphaCharCountTaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_count(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_count.data()));
    taskDataSeq->outputs_count.emplace_back(reference_count.size());

    muradov_m_count_alpha_chars_mpi::AlphaCharCountTaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_count[0], global_count[0]);
  }
}

void run_test_for_string(const std::string& test_str) {
  boost::mpi::communicator world;
  std::vector<int32_t> global_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int expected_alpha_count = 0;

  if (world.rank() == 0) {
    expected_alpha_count = std::count_if(test_str.begin(), test_str.end(),
                                         [](char c) { return std::isalpha(static_cast<unsigned char>(c)); });

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(test_str.data())));
    taskDataPar->inputs_count.emplace_back(test_str.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }

  muradov_m_count_alpha_chars_mpi::AlphaCharCountTaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_count(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(test_str.data())));
    taskDataSeq->inputs_count.emplace_back(test_str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_count.data()));
    taskDataSeq->outputs_count.emplace_back(reference_count.size());

    muradov_m_count_alpha_chars_mpi::AlphaCharCountTaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_count[0], global_count[0]);
    ASSERT_EQ(expected_alpha_count, global_count[0]);
  }
}

TEST(muradov_m_count_alpha_chars_mpi, test_various_lengths) {
  run_test_for_string("");
  run_test_for_string("A");
  run_test_for_string("Ab");
  run_test_for_string("A1b");
  run_test_for_string("A1bC2");
  run_test_for_string("A1bC2d3");
  run_test_for_string("AbCdeFg");
  run_test_for_string("A1b2C3d4e");
}