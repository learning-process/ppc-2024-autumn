// Copyright 2023 Konkov Ivan
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <string>
#include <vector>

#include "mpi/konkov_i_count_words/include/ops_mpi.hpp"

std::string generate_random_string(int length) {
  static const char alphanum[] =
      "0123456789"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz";
  std::string result;
  result.reserve(length);
  for (int i = 0; i < length; ++i) {
    result += alphanum[rand() % (sizeof(alphanum) - 1)];
  }
  return result;
}

TEST(konkov_i_count_words_mpi, Test_Empty_String) {
  boost::mpi::communicator world;
  std::string input;
  int expected_count = 0;

  std::vector<int> out(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  konkov_i_count_words_mpi::CountWordsTaskParallel testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected_count, out[0]);
  }
}

TEST(konkov_i_count_words_mpi, Test_Single_Word) {
  boost::mpi::communicator world;
  std::string input = "Hello";
  int expected_count = 1;

  std::vector<int> out(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  konkov_i_count_words_mpi::CountWordsTaskParallel testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected_count, out[0]);
  }
}

TEST(konkov_i_count_words_mpi, Test_Multiple_Words) {
  boost::mpi::communicator world;
  std::string input = "Hello world this is a test";
  int expected_count = 6;

  std::vector<int> out(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  konkov_i_count_words_mpi::CountWordsTaskParallel testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected_count, out[0]);
  }
}

TEST(konkov_i_count_words_mpi, Test_Random_String) {
  boost::mpi::communicator world;
  std::string input = generate_random_string(100);

  std::istringstream stream(input);
  std::string word;
  int expected_count = 0;
  while (stream >> word) {
    expected_count++;
  }

  std::vector<int> out(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  konkov_i_count_words_mpi::CountWordsTaskParallel testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected_count, out[0]);
  }
}

TEST(konkov_i_count_words_mpi, Test_Multiple_Spaces) {
  boost::mpi::communicator world;
  std::string input = "Hello   world   this   is   a   test";
  int expected_count = 6;

  std::vector<int> out(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  konkov_i_count_words_mpi::CountWordsTaskParallel testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected_count, out[0]);
  }
}

TEST(konkov_i_count_words_mpi, Test_Newlines) {
  boost::mpi::communicator world;
  std::string input = "Hello\nworld\nthis\nis\na\ntest";
  int expected_count = 6;

  std::vector<int> out(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  konkov_i_count_words_mpi::CountWordsTaskParallel testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected_count, out[0]);
  }
}

TEST(konkov_i_count_words_mpi, Test_Punctuation) {
  boost::mpi::communicator world;
  std::string input = "Hello, world! This is a test.";
  int expected_count = 6;

  std::vector<int> out(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  konkov_i_count_words_mpi::CountWordsTaskParallel testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected_count, out[0]);
  }
}