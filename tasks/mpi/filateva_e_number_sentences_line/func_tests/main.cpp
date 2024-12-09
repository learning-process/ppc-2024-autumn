// Filateva Elizaveta Number_of_sentences_per_line
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/filateva_e_number_sentences_line/include/ops_mpi.hpp"

std::string getRandomLine(int max_count) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::string line = "Hello world. How many words are in this sentence? The task of parallel programming!";
  int count = gen() % max_count;
  for (int i = 0; i < count; ++i) {
    line += line;
  }
  return line;
}

TEST(filateva_e_number_sentences_line_mpi, Test_countSentences) {
  std::string line = "Hello world. How many words are in this sentence? The task of parallel programming!";
  int count = filateva_e_number_sentences_line_mpi::countSentences(line);
  ASSERT_EQ(3, count);
}

TEST(filateva_e_number_sentences_line_mpi, one_sentence_line_1) {
  boost::mpi::communicator world;
  std::string line = "Hello world.";
  std::vector<int> out(1, 0);
  // // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(line.data()));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  filateva_e_number_sentences_line_mpi::NumberSentencesLineParallel NumS(taskDataPar);
  ASSERT_EQ(NumS.validation(), true);
  NumS.pre_processing();
  NumS.run();
  NumS.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(1, out[0]);
  }
}

TEST(filateva_e_number_sentences_line_mpi, one_sentence_line_2) {
  boost::mpi::communicator world;
  std::string line = "Hello world";
  std::vector<int> out(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(line.data()));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  filateva_e_number_sentences_line_mpi::NumberSentencesLineParallel NumS(taskDataPar);
  ASSERT_EQ(NumS.validation(), true);
  NumS.pre_processing();
  NumS.run();
  NumS.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(1, out[0]);
  }
}

TEST(filateva_e_number_sentences_line_mpi, one_sentence_line_3) {
  boost::mpi::communicator world;
  std::string line = "Hello world!";
  std::vector<int> out(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(line.data()));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  filateva_e_number_sentences_line_mpi::NumberSentencesLineParallel NumS(taskDataPar);
  ASSERT_EQ(NumS.validation(), true);
  NumS.pre_processing();
  NumS.run();
  NumS.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(1, out[0]);
  }
}

TEST(filateva_e_number_sentences_line_mpi, one_sentence_line_4) {
  boost::mpi::communicator world;
  std::string line = "Hello world?";
  std::vector<int> out(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(line.data()));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  filateva_e_number_sentences_line_mpi::NumberSentencesLineParallel NumS(taskDataPar);
  ASSERT_EQ(NumS.validation(), true);
  NumS.pre_processing();
  NumS.run();
  NumS.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(1, out[0]);
  }
}

TEST(filateva_e_number_sentences_line_mpi, empty_string) {
  boost::mpi::communicator world;
  std::string line;
  std::vector<int> out(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(line.data()));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  filateva_e_number_sentences_line_mpi::NumberSentencesLineParallel NumS(taskDataPar);
  ASSERT_EQ(NumS.validation(), true);
  NumS.pre_processing();
  NumS.run();
  NumS.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(0, out[0]);
  }
}

TEST(filateva_e_number_sentences_line_mpi, random_text_1) {
  boost::mpi::communicator world;
  std::string line;
  std::vector<int> out(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    line = getRandomLine(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(line.data()));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  filateva_e_number_sentences_line_mpi::NumberSentencesLineParallel NumS(taskDataPar);
  ASSERT_EQ(NumS.validation(), true);
  NumS.pre_processing();
  NumS.run();
  NumS.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> ref_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(line.data()));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_out.data()));
    taskDataSeq->outputs_count.emplace_back(ref_out.size());

    // Create Task
    filateva_e_number_sentences_line_mpi::NumberSentencesLineSequential NumSSeq(taskDataSeq);
    ASSERT_EQ(NumSSeq.validation(), true);
    NumSSeq.pre_processing();
    NumSSeq.run();
    NumSSeq.post_processing();

    ASSERT_EQ(out[0], ref_out[0]);
  }
}

TEST(filateva_e_number_sentences_line_mpi, random_text_2) {
  boost::mpi::communicator world;
  std::string line;
  std::vector<int> out(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    line = getRandomLine(3);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(line.data()));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  filateva_e_number_sentences_line_mpi::NumberSentencesLineParallel NumS(taskDataPar);
  ASSERT_EQ(NumS.validation(), true);
  NumS.pre_processing();
  NumS.run();
  NumS.post_processing();

  if (world.rank() == 0) {
    // // Create data
    std::vector<int> ref_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(line.data()));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_out.data()));
    taskDataSeq->outputs_count.emplace_back(ref_out.size());

    // Create Task
    filateva_e_number_sentences_line_mpi::NumberSentencesLineSequential NumSSeq(taskDataSeq);
    ASSERT_EQ(NumSSeq.validation(), true);
    NumSSeq.pre_processing();
    NumSSeq.run();
    NumSSeq.post_processing();

    ASSERT_EQ(out[0], ref_out[0]);
  }
}

TEST(filateva_e_number_sentences_line_mpi, random_text_3) {
  boost::mpi::communicator world;
  std::string line;
  std::vector<int> out(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    line = getRandomLine(5);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(line.data()));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  filateva_e_number_sentences_line_mpi::NumberSentencesLineParallel NumS(taskDataPar);
  ASSERT_EQ(NumS.validation(), true);
  NumS.pre_processing();
  NumS.run();
  NumS.post_processing();

  if (world.rank() == 0) {
    // // Create data
    std::vector<int> ref_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(line.data()));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_out.data()));
    taskDataSeq->outputs_count.emplace_back(ref_out.size());

    // Create Task
    filateva_e_number_sentences_line_mpi::NumberSentencesLineSequential NumSSeq(taskDataSeq);
    ASSERT_EQ(NumSSeq.validation(), true);
    NumSSeq.pre_processing();
    NumSSeq.run();
    NumSSeq.post_processing();

    ASSERT_EQ(out[0], ref_out[0]);
  }
}

TEST(filateva_e_number_sentences_line_mpi, random_text_4) {
  boost::mpi::communicator world;
  std::string line;
  std::vector<int> out(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    line = getRandomLine(10);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(line.data()));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  filateva_e_number_sentences_line_mpi::NumberSentencesLineParallel NumS(taskDataPar);
  ASSERT_EQ(NumS.validation(), true);
  NumS.pre_processing();
  NumS.run();
  NumS.post_processing();

  if (world.rank() == 0) {
    // // Create data
    std::vector<int> ref_out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(line.data()));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_out.data()));
    taskDataSeq->outputs_count.emplace_back(ref_out.size());

    // Create Task
    filateva_e_number_sentences_line_mpi::NumberSentencesLineSequential NumSSeq(taskDataSeq);
    ASSERT_EQ(NumSSeq.validation(), true);
    NumSSeq.pre_processing();
    NumSSeq.run();
    NumSSeq.post_processing();

    ASSERT_EQ(out[0], ref_out[0]);
  }
}

TEST(filateva_e_number_sentences_line_mpi, sentence_without_dot) {
  boost::mpi::communicator world;
  std::string line = "Hello world. Hello world! Hello world";
  std::vector<int> out(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(line.data()));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  filateva_e_number_sentences_line_mpi::NumberSentencesLineParallel NumS(taskDataPar);
  ASSERT_EQ(NumS.validation(), true);
  NumS.pre_processing();
  NumS.run();
  NumS.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(3, out[0]);
  }
}