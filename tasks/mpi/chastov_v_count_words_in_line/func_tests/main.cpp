// Copyright 2024 Chastov Vyacheslav
#include <gtest/gtest.h>

#include "mpi/chastov_v_count_words_in_line/include/ops_mpi.hpp"
#include "mpi/chastov_v_count_words_in_line/src/ops_mpi.cpp"

// Test to check the behavior of the MPI word counting function with an empty string
TEST(chastov_v_count_words_in_line_mpi, empty_string) {
  boost::mpi::communicator world;
  std::vector<char> input = {};
  std::vector<int> wordsFound(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(wordsFound.data()));
    taskDataPar->outputs_count.emplace_back(wordsFound.size());

    chastov_v_count_words_in_line_mpi::TestMPITaskParallel testTaskParallel(taskDataPar);
    ASSERT_FALSE(testTaskParallel.validation());
  }
}

// Test to verify the MPI word counting function with a single word input ("hello")
TEST(chastov_v_count_words_in_line_mpi, words_1) {
  boost::mpi::communicator world;
  std::vector<char> input;
  std::string testString = "hello";
  for (unsigned long int j = 0; j < testString.length(); j++) {
    input.push_back(testString[j]);
  }
  std::vector<int> wordsFound(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(wordsFound.data()));
    taskDataPar->outputs_count.emplace_back(wordsFound.size());
  }

  chastov_v_count_words_in_line_mpi::TestMPITaskParallel testTaskParallel(taskDataPar);
  ASSERT_TRUE(testTaskParallel.validation());
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> referenceWordFound(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
    taskDataSequential->inputs_count.emplace_back(input.size());
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(referenceWordFound.data()));
    taskDataSequential->outputs_count.emplace_back(referenceWordFound.size());

    // Create Task
    chastov_v_count_words_in_line_mpi::TestMPITaskSequential testTaskSequential(taskDataSequential);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(wordsFound[0], referenceWordFound[0]);
  }
}

// Test to verify the MPI word counting function with an input string containing four words ("My name is Slava")
TEST(chastov_v_count_words_in_line_mpi, words_4) {
  boost::mpi::communicator world;
  std::vector<char> input;
  std::string testString = "My name is Slava";
  for (unsigned long int j = 0; j < testString.length(); j++) {
    input.push_back(testString[j]);
  }
  std::vector<int> wordsFound(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(wordsFound.data()));
    taskDataPar->outputs_count.emplace_back(wordsFound.size());
  }

  chastov_v_count_words_in_line_mpi::TestMPITaskParallel testTaskParallel(taskDataPar);
  ASSERT_TRUE(testTaskParallel.validation());
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> referenceWordFound(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
    taskDataSequential->inputs_count.emplace_back(input.size());
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(referenceWordFound.data()));
    taskDataSequential->outputs_count.emplace_back(referenceWordFound.size());

    // Create Task
    chastov_v_count_words_in_line_mpi::TestMPITaskSequential testTaskSequential(taskDataSequential);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(wordsFound[0], referenceWordFound[0]);
  }
}

// Test to verify the MPI word counting function with an input string that generates 450 words
TEST(chastov_v_count_words_in_line_mpi, words_300) {
  boost::mpi::communicator world;
  std::vector<char> input = createTestInput(20);
  std::vector<int> wordsFound(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(wordsFound.data()));
    taskDataPar->outputs_count.emplace_back(wordsFound.size());
  }

  chastov_v_count_words_in_line_mpi::TestMPITaskParallel testTaskParallel(taskDataPar);
  ASSERT_TRUE(testTaskParallel.validation());
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> referenceWordFound(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
    taskDataSequential->inputs_count.emplace_back(input.size());
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(referenceWordFound.data()));
    taskDataSequential->outputs_count.emplace_back(referenceWordFound.size());

    // Create Task
    chastov_v_count_words_in_line_mpi::TestMPITaskSequential testTaskSequential(taskDataSequential);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(wordsFound[0], referenceWordFound[0]);
  }
}

// Test to verify the MPI word counting function with an input string that generates 1500 words
TEST(chastov_v_count_words_in_line_mpi, words_1500) {
  boost::mpi::communicator world;
  std::vector<char> input = createTestInput(100);
  std::vector<int> wordsFound(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(wordsFound.data()));
    taskDataPar->outputs_count.emplace_back(wordsFound.size());
  }

  chastov_v_count_words_in_line_mpi::TestMPITaskParallel testTaskParallel(taskDataPar);
  ASSERT_TRUE(testTaskParallel.validation());
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> referenceWordFound(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
    taskDataSequential->inputs_count.emplace_back(input.size());
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(referenceWordFound.data()));
    taskDataSequential->outputs_count.emplace_back(referenceWordFound.size());

    // Create Task
    chastov_v_count_words_in_line_mpi::TestMPITaskSequential testTaskSequential(taskDataSequential);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(wordsFound[0], referenceWordFound[0]);
  }
}

// Test to verify the MPI word counting function with an input string that generates 7500 words
TEST(chastov_v_count_words_in_line_mpi, words_7500) {
  boost::mpi::communicator world;
  std::vector<char> input = createTestInput(500);
  std::vector<int> wordsFound(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(wordsFound.data()));
    taskDataPar->outputs_count.emplace_back(wordsFound.size());
  }

  chastov_v_count_words_in_line_mpi::TestMPITaskParallel testTaskParallel(taskDataPar);
  ASSERT_TRUE(testTaskParallel.validation());
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> referenceWordFound(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
    taskDataSequential->inputs_count.emplace_back(input.size());
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(referenceWordFound.data()));
    taskDataSequential->outputs_count.emplace_back(referenceWordFound.size());

    // Create Task
    chastov_v_count_words_in_line_mpi::TestMPITaskSequential testTaskSequential(taskDataSequential);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(wordsFound[0], referenceWordFound[0]);
  }
}

// The test tests the functionality of counting words in a string with many spaces between words
TEST(chastov_v_count_words_in_line_mpi, multiple_spaces) {
  boost::mpi::communicator world;
  std::vector<char> input = {'T', 'h', 'i', 's', ' ', 'i', 's', ' ', 'a', ' ', 't', 'e', 's', 't'};
  std::vector<int> wordsFound(1, 0);
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(wordsFound.data()));
    taskDataPar->outputs_count.emplace_back(wordsFound.size());
  }

  auto testTaskParallel = std::make_shared<chastov_v_count_words_in_line_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_TRUE(testTaskParallel->validation());
  testTaskParallel->pre_processing();
  testTaskParallel->run();
  testTaskParallel->post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(wordsFound[0], 4);
  }
}

// Test checks the word count in a string with multiple spaces between words
TEST(chastov_v_count_words_in_line_mpi, multiple_consecutive_spaces) {
  boost::mpi::communicator world;
  std::vector<char> input;
  std::string testString = "Hello   world   MPI";
  for (unsigned long int j = 0; j < testString.length(); j++) {
    input.push_back(testString[j]);
  }
  std::vector<int> wordsFound(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(wordsFound.data()));
    taskDataPar->outputs_count.emplace_back(wordsFound.size());
  }

  chastov_v_count_words_in_line_mpi::TestMPITaskParallel testTaskParallel(taskDataPar);
  ASSERT_TRUE(testTaskParallel.validation());
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> referenceWordFound(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
    taskDataSequential->inputs_count.emplace_back(input.size());
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(referenceWordFound.data()));
    taskDataSequential->outputs_count.emplace_back(referenceWordFound.size());

    // Create Task
    chastov_v_count_words_in_line_mpi::TestMPITaskSequential testTaskSequential(taskDataSequential);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(wordsFound[0], referenceWordFound[0]);
  }
}