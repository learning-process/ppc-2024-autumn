#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <memory>
#include <string>
#include <vector>

#include "mpi/tyurin_m_count_sentences_in_string/include/ops_mpi.hpp"

TEST(tyurin_m_count_sentences_in_string_mpi, test_all_sentence_endings) {
  boost::mpi::communicator world;
  std::string input_str;
  std::vector<int32_t> global_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    input_str = "Hello world! How are you? I am fine.";
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_str));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testMpiTaskParallel =
      std::make_shared<tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_count(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_str));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_count.data()));
    taskDataSeq->outputs_count.emplace_back(1);

    auto testMpiTaskSequential =
        std::make_shared<tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskSequential>(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential->validation(), true);
    testMpiTaskSequential->pre_processing();
    testMpiTaskSequential->run();
    testMpiTaskSequential->post_processing();

    ASSERT_EQ(reference_count[0], global_count[0]);
  }
}

TEST(tyurin_m_count_sentences_in_string_mpi, test_no_sentence_endings) {
  boost::mpi::communicator world;
  std::string input_str;
  std::vector<int32_t> global_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    input_str = "This is a test without sentence endings";
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_str));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testMpiTaskParallel =
      std::make_shared<tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_count(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_str));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_count.data()));
    taskDataSeq->outputs_count.emplace_back(1);

    auto testMpiTaskSequential =
        std::make_shared<tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskSequential>(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential->validation(), true);
    testMpiTaskSequential->pre_processing();
    testMpiTaskSequential->run();
    testMpiTaskSequential->post_processing();

    ASSERT_EQ(reference_count[0], global_count[0]);
  }
}

TEST(tyurin_m_count_sentences_in_string_mpi, test_mixed_content) {
  boost::mpi::communicator world;
  std::string input_str;
  std::vector<int32_t> global_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    input_str = "Sentence one. Another sentence! And another one? And one more.";
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_str));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testMpiTaskParallel =
      std::make_shared<tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_count(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_str));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_count.data()));
    taskDataSeq->outputs_count.emplace_back(1);

    auto testMpiTaskSequential =
        std::make_shared<tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskSequential>(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential->validation(), true);
    testMpiTaskSequential->pre_processing();
    testMpiTaskSequential->run();
    testMpiTaskSequential->post_processing();

    ASSERT_EQ(reference_count[0], global_count[0]);
  }
}

TEST(tyurin_m_count_sentences_in_string_mpi, test_empty_string) {
  boost::mpi::communicator world;
  std::string input_str;
  std::vector<int32_t> global_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    input_str = "";
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_str));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testMpiTaskParallel =
      std::make_shared<tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_count(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_str));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_count.data()));
    taskDataSeq->outputs_count.emplace_back(1);

    auto testMpiTaskSequential =
        std::make_shared<tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskSequential>(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential->validation(), true);
    testMpiTaskSequential->pre_processing();
    testMpiTaskSequential->run();
    testMpiTaskSequential->post_processing();

    ASSERT_EQ(reference_count[0], global_count[0]);
  }
}

TEST(tyurin_m_count_sentences_in_string_mpi, test_multiple_consecutive_endings) {
  boost::mpi::communicator world;
  std::string input_str;
  std::vector<int32_t> global_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    input_str = "First sentence. Second sentence?! Third sentence.";
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_str));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testMpiTaskParallel =
      std::make_shared<tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_count(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_str));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_count.data()));
    taskDataSeq->outputs_count.emplace_back(1);

    auto testMpiTaskSequential =
        std::make_shared<tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskSequential>(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential->validation(), true);
    testMpiTaskSequential->pre_processing();
    testMpiTaskSequential->run();
    testMpiTaskSequential->post_processing();

    ASSERT_EQ(reference_count[0], global_count[0]);
  }
}

TEST(tyurin_m_count_sentences_in_string_mpi, test_various_whitespaces_between_sentences) {
  boost::mpi::communicator world;
  std::string input_str;
  std::vector<int32_t> global_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    input_str = "Sentence one.   \nSecond sentence!\tThird sentence?";
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_str));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testMpiTaskParallel =
      std::make_shared<tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_count(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_str));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_count.data()));
    taskDataSeq->outputs_count.emplace_back(1);

    auto testMpiTaskSequential =
        std::make_shared<tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskSequential>(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential->validation(), true);
    testMpiTaskSequential->pre_processing();
    testMpiTaskSequential->run();
    testMpiTaskSequential->post_processing();

    ASSERT_EQ(reference_count[0], global_count[0]);
  }
}
