#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <string>
#include <vector>

#include "mpi/shkurinskaya_e_count_sentences/include/ops_mpi.hpp"

TEST(shkurinskaya_e_count_sentences_mpi, Test_Random_String) {
  boost::mpi::communicator world;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> char_dist(32, 126);
  std::string input_text;
  int string_length = 1000;
  for (int i = 0; i < string_length; ++i) {
    input_text += static_cast<char>(char_dist(gen));
  }
  std::vector<int> global_result(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_text));
    taskDataPar->inputs_count.emplace_back(input_text.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }
  shkurinskaya_e_count_sentences_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int> reference_result(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_text));
    taskDataSeq->inputs_count.emplace_back(input_text.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    shkurinskaya_e_count_sentences_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(global_result[0], reference_result[0]);
  }
}

TEST(shkurinskaya_e_count_sentences_mpi, Test_Count_Sentences) {
  boost::mpi::communicator world;
  std::string input_text;
  std::vector<int> global_result(1, 0);

  // Create TaskData for parallel execution
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input_text = "Hello world! This is a test. Let's see if it works?";
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_text));
    taskDataPar->inputs_count.emplace_back(input_text.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  shkurinskaya_e_count_sentences_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create TaskData for sequential execution
    std::vector<int> reference_result(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_text));
    taskDataSeq->inputs_count.emplace_back(input_text.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    shkurinskaya_e_count_sentences_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_result[0], reference_result[0]);
  }
}

TEST(shkurinskaya_e_count_sentences_mpi, Test_Empty_Text) {
  boost::mpi::communicator world;
  std::string input_text;
  std::vector<int> global_result(1, 0);

  // Create TaskData for parallel execution
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_text));
    taskDataPar->inputs_count.emplace_back(input_text.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  shkurinskaya_e_count_sentences_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create TaskData for sequential execution
    std::vector<int> reference_result(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_text));
    taskDataSeq->inputs_count.emplace_back(input_text.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    shkurinskaya_e_count_sentences_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_result[0], reference_result[0]);
  }
}

TEST(shkurinskaya_e_count_sentences_mpi, Test_Multiple_Endings) {
  boost::mpi::communicator world;
  std::string input_text;
  std::vector<int> global_result(1, 0);

  // Create TaskData for parallel execution
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input_text = "Wow!! Really?! That's amazing...";
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_text));
    taskDataPar->inputs_count.emplace_back(input_text.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  shkurinskaya_e_count_sentences_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create TaskData for sequential execution
    std::vector<int> reference_result(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_text));
    taskDataSeq->inputs_count.emplace_back(input_text.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    shkurinskaya_e_count_sentences_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_result[0], reference_result[0]);
  }
}

TEST(shkurinskaya_e_count_sentences_mpi, Test_Single_Sentence) {
  boost::mpi::communicator world;
  std::string input_text;
  std::vector<int> global_result(1, 0);

  // Create TaskData for parallel execution
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input_text = "This is just one sentence.";
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_text));
    taskDataPar->inputs_count.emplace_back(input_text.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  shkurinskaya_e_count_sentences_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create TaskData for sequential execution
    std::vector<int> reference_result(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_text));
    taskDataSeq->inputs_count.emplace_back(input_text.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    shkurinskaya_e_count_sentences_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_result[0], reference_result[0]);
  }
}

TEST(shkurinskaya_e_count_sentences_mpi, Test_No_Punctuation) {
  boost::mpi::communicator world;
  std::string input_text;
  std::vector<int> global_result(1, 0);

  // Create TaskData for parallel execution
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input_text = "This text has no punctuation and therefore should count as zero sentences";
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_text));
    taskDataPar->inputs_count.emplace_back(input_text.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  shkurinskaya_e_count_sentences_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create TaskData for sequential execution
    std::vector<int> reference_result(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_text));
    taskDataSeq->inputs_count.emplace_back(input_text.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    shkurinskaya_e_count_sentences_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_result[0], reference_result[0]);
  }
}
