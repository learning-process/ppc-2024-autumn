#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/guseynov_e_check_lex_order_of_two_string/include/ops_mpi.hpp"

TEST(guseynov_e_check_lex_order_of_two_string_mpi, Test_equal_words) {
  boost::mpi::communicator world;
  std::vector<std::vector<char>> global_vec(2, std::vector<char>(120, 'a'));
  std::vector<int32_t> global_res(1, -1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec[1].data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->inputs_count.emplace_back(global_vec[0].size());
    taskDataPar->inputs_count.emplace_back(global_vec[1].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  guseynov_e_check_lex_order_of_two_string_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_res(1, -1);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec[0].data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec[1].data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->inputs_count.emplace_back(global_vec[0].size());
    taskDataSeq->inputs_count.emplace_back(global_vec[1].size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // create Task
    guseynov_e_check_lex_order_of_two_string_mpi::TestMPITaskSequential testMPITaskSequantial(taskDataSeq);
    ASSERT_EQ(testMPITaskSequantial.validation(), true);
    testMPITaskSequantial.pre_processing();
    testMPITaskSequantial.run();
    testMPITaskSequantial.post_processing();
    ASSERT_EQ(reference_res[0], global_res[0]);
  }
}

TEST(guseynov_e_check_lex_order_of_two_string_mpi, Test_second_string_is_greater) {
  boost::mpi::communicator world;
  std::vector<std::vector<char>> global_vec(2, std::vector<char>(240, 'a'));
  global_vec[1][239] = 'b';
  std::vector<int32_t> global_res(1, -1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec[1].data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->inputs_count.emplace_back(global_vec[0].size());
    taskDataPar->inputs_count.emplace_back(global_vec[1].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  guseynov_e_check_lex_order_of_two_string_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_res(1, -1);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec[0].data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec[1].data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->inputs_count.emplace_back(global_vec[0].size());
    taskDataSeq->inputs_count.emplace_back(global_vec[1].size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // create Task
    guseynov_e_check_lex_order_of_two_string_mpi::TestMPITaskSequential testMPITaskSequantial(taskDataSeq);
    ASSERT_EQ(testMPITaskSequantial.validation(), true);
    testMPITaskSequantial.pre_processing();
    testMPITaskSequantial.run();
    testMPITaskSequantial.post_processing();
    ASSERT_EQ(reference_res[0], global_res[0]);
  }
}

TEST(guseynov_e_check_lex_order_of_two_string_mpi, Test_first_string_is_greater) {
  boost::mpi::communicator world;
  std::vector<std::vector<char>> global_vec(2, std::vector<char>(240, 'a'));
  global_vec[0][0] = 'b';
  std::vector<int32_t> global_res(1, -1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec[1].data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->inputs_count.emplace_back(global_vec[0].size());
    taskDataPar->inputs_count.emplace_back(global_vec[1].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  guseynov_e_check_lex_order_of_two_string_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_res(1, -1);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec[0].data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec[1].data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->inputs_count.emplace_back(global_vec[0].size());
    taskDataSeq->inputs_count.emplace_back(global_vec[1].size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // create Task
    guseynov_e_check_lex_order_of_two_string_mpi::TestMPITaskSequential testMPITaskSequantial(taskDataSeq);
    ASSERT_EQ(testMPITaskSequantial.validation(), true);
    testMPITaskSequantial.pre_processing();
    testMPITaskSequantial.run();
    testMPITaskSequantial.post_processing();
    ASSERT_EQ(reference_res[0], global_res[0]);
  }
}

TEST(guseynov_e_check_lex_order_of_two_string_mpi, Test_first_string_is_prefix) {
  boost::mpi::communicator world;
  std::vector<std::vector<char>> global_vec(2, std::vector<char>(360, 'a'));
  global_vec[1].push_back('b');
  std::vector<int32_t> global_res(1, -1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec[1].data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->inputs_count.emplace_back(global_vec[0].size());
    taskDataPar->inputs_count.emplace_back(global_vec[1].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  guseynov_e_check_lex_order_of_two_string_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_res(1, -1);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec[0].data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec[1].data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->inputs_count.emplace_back(global_vec[0].size());
    taskDataSeq->inputs_count.emplace_back(global_vec[1].size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // create Task
    guseynov_e_check_lex_order_of_two_string_mpi::TestMPITaskSequential testMPITaskSequantial(taskDataSeq);
    ASSERT_EQ(testMPITaskSequantial.validation(), true);
    testMPITaskSequantial.pre_processing();
    testMPITaskSequantial.run();
    testMPITaskSequantial.post_processing();
    ASSERT_EQ(reference_res[0], global_res[0]);
  }
}

TEST(guseynov_e_check_lex_order_of_two_string_mpi, Test_second_string_is_prefix) {
  boost::mpi::communicator world;
  std::vector<std::vector<char>> global_vec(2, std::vector<char>(360, 'a'));
  global_vec[0].push_back('b');
  std::vector<int32_t> global_res(1, -1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec[1].data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->inputs_count.emplace_back(global_vec[0].size());
    taskDataPar->inputs_count.emplace_back(global_vec[1].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  guseynov_e_check_lex_order_of_two_string_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_res(1, -1);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec[0].data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec[1].data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->inputs_count.emplace_back(global_vec[0].size());
    taskDataSeq->inputs_count.emplace_back(global_vec[1].size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    // create Task
    guseynov_e_check_lex_order_of_two_string_mpi::TestMPITaskSequential testMPITaskSequantial(taskDataSeq);
    ASSERT_EQ(testMPITaskSequantial.validation(), true);
    testMPITaskSequantial.pre_processing();
    testMPITaskSequantial.run();
    testMPITaskSequantial.post_processing();
    ASSERT_EQ(reference_res[0], global_res[0]);
  }
}