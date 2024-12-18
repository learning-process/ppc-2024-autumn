// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/kolodkin_g_sentence_count/include/ops_mpi.hpp"

TEST(kolodkin_g_sentence_count, Test_empty_string) {
  boost::mpi::communicator world;
  std::vector<char> global_str;

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataMpi->inputs_count.emplace_back(global_str.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }

  // Create Task
  kolodkin_g_sentence_count_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    kolodkin_g_sentence_count_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
    ASSERT_EQ(reference_out[0], 0);
  }
}

TEST(kolodkin_g_sentence_count, Test_two_sentences) {
  boost::mpi::communicator world;
  std::vector<char> global_str;

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::string str = "Hello!My name is Grisha!";
    for (unsigned long int i = 0; i < str.length(); i++) {
      global_str.push_back(str[i]);
    }
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataMpi->inputs_count.emplace_back(global_str.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }

  // Create Task
  kolodkin_g_sentence_count_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    kolodkin_g_sentence_count_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
    ASSERT_EQ(reference_out[0], 2);
  }
}

TEST(kolodkin_g_sentence_count, Test_sentences_with_special_symbols) {
  boost::mpi::communicator world;
  std::vector<char> global_str;

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::string str = "Hello! My name is Grisha! I have cat,dog,parrot.";
    for (unsigned long int i = 0; i < str.length(); i++) {
      global_str.push_back(str[i]);
    }
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataMpi->inputs_count.emplace_back(global_str.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }

  // Create Task
  kolodkin_g_sentence_count_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    kolodkin_g_sentence_count_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
    ASSERT_EQ(reference_out[0], 3);
  }
}

TEST(kolodkin_g_sentence_count, Test_sentences_with_special_symbols_in_end_of_sentence) {
  boost::mpi::communicator world;
  std::vector<char> global_str;

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::string str = "Hello! My name is Grisha! I have cat, dog, parrot. What is your name? How are you? Well...";
    for (unsigned long int i = 0; i < str.length(); i++) {
      global_str.push_back(str[i]);
    }
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataMpi->inputs_count.emplace_back(global_str.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }

  // Create Task
  kolodkin_g_sentence_count_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    kolodkin_g_sentence_count_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
    ASSERT_EQ(reference_out[0], 6);
  }
}
TEST(kolodkin_g_sentence_count, Test_sentences_with_double_symbols) {
  boost::mpi::communicator world;
  std::vector<char> global_str;

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::string str =
        "Hello!! My name is Grisha!! I have two pets: cat,dog,parrot. What is your name?! How are you!? Well...";
    for (unsigned long int i = 0; i < str.length(); i++) {
      global_str.push_back(str[i]);
    }
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataMpi->inputs_count.emplace_back(global_str.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }

  // Create Task
  kolodkin_g_sentence_count_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    kolodkin_g_sentence_count_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
    ASSERT_EQ(reference_out[0], 6);
  }
}
TEST(kolodkin_g_sentence_count, Big_text) {
  boost::mpi::communicator world;
  std::vector<char> global_str;

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::string str =
        "Otche nash, ize esi na nebeseh! Da svytitsa imya tvoe, da priidet tsarstvo tvoe! Da budet volya tvoya, ako na "
        "nebeseh i na zemle. Hleb nas nasyshnii dazd nam dnes, i ostavi nam dolgi nasha. Yakozhe i my ostavlyaem "
        "dolznikom nashim! I ne vvedi nas vo iskushenie, no izbavi nas ot lukavogo... Amin!";
    for (unsigned long int i = 0; i < str.length(); i++) {
      global_str.push_back(str[i]);
    }
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataMpi->inputs_count.emplace_back(global_str.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }

  // Create Task
  kolodkin_g_sentence_count_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> reference_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out.size());

    // Create Task
    kolodkin_g_sentence_count_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(reference_out[0], global_out[0]);
    ASSERT_EQ(reference_out[0], 7);
  }
}
