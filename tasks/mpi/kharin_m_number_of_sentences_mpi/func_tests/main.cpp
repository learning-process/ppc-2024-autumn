#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>

#include "mpi/kharin_m_number_of_sentences_mpi/include/ops_mpi.hpp"

TEST(Parallel_Sentences_Count_MPI, Test_Simple_Sentences) {
  boost::mpi::communicator world;
  std::string input_text;
  std::vector<int> sentence_count(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  input_text = "This is sentence one. This is sentence two! Is this sentence three? This is sentence four.";

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input_text.c_str())));
  taskDataPar->inputs_count.emplace_back(input_text.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(sentence_count.data()));
  taskDataPar->outputs_count.emplace_back(sentence_count.size());

  kharin_m_number_of_sentences_mpi::CountSentencesParallel countSentencesParallel(taskDataPar);
  ASSERT_EQ(countSentencesParallel.validation(), true);
  countSentencesParallel.pre_processing();
  countSentencesParallel.run();
  countSentencesParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_count(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input_text.c_str())));
    taskDataSeq->inputs_count.emplace_back(input_text.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_count.data()));
    taskDataSeq->outputs_count.emplace_back(reference_count.size());

    // Run sequential version
    kharin_m_number_of_sentences_mpi::CountSentencesSequential countSentencesSequential(taskDataSeq);
    ASSERT_EQ(countSentencesSequential.validation(), true);
    countSentencesSequential.pre_processing();
    countSentencesSequential.run();
    countSentencesSequential.post_processing();

    // Compare results
    ASSERT_EQ(reference_count[0], 4);
    ASSERT_EQ(reference_count[0], sentence_count[0]);
  }
}

TEST(Parallel_Sentences_Count_MPI, Test_Empty_Text) {
  boost::mpi::communicator world;
  std::string input_text;
  std::vector<int> sentence_count(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input_text.c_str())));
  taskDataPar->inputs_count.emplace_back(input_text.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(sentence_count.data()));
  taskDataPar->outputs_count.emplace_back(sentence_count.size());

  kharin_m_number_of_sentences_mpi::CountSentencesParallel countSentencesParallel(taskDataPar);
  ASSERT_EQ(countSentencesParallel.validation(), true);
  countSentencesParallel.pre_processing();
  countSentencesParallel.run();
  countSentencesParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_count(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input_text.c_str())));
    taskDataSeq->inputs_count.emplace_back(input_text.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_count.data()));
    taskDataSeq->outputs_count.emplace_back(reference_count.size());

    // Run sequential version
    kharin_m_number_of_sentences_mpi::CountSentencesSequential countSentencesSequential(taskDataSeq);
    ASSERT_EQ(countSentencesSequential.validation(), true);
    countSentencesSequential.pre_processing();
    countSentencesSequential.run();
    countSentencesSequential.post_processing();

    // Compare results
    ASSERT_EQ(reference_count[0], 0);
    ASSERT_EQ(reference_count[0], sentence_count[0]);
  }
}

TEST(Parallel_Sentences_Count_MPI, Test_Long_Text) {
  boost::mpi::communicator world;
  std::string input_text;
  std::vector<int> sentence_count(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  for (int i = 0; i < 100; i++) {
    input_text += "This is sentence number " + std::to_string(i + 1) + ". ";
  }

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input_text.c_str())));
  taskDataPar->inputs_count.emplace_back(input_text.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(sentence_count.data()));
  taskDataPar->outputs_count.emplace_back(sentence_count.size());

  kharin_m_number_of_sentences_mpi::CountSentencesParallel countSentencesParallel(taskDataPar);
  ASSERT_EQ(countSentencesParallel.validation(), true);
  countSentencesParallel.pre_processing();
  countSentencesParallel.run();
  countSentencesParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_count(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input_text.c_str())));
    taskDataSeq->inputs_count.emplace_back(input_text.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_count.data()));
    taskDataSeq->outputs_count.emplace_back(reference_count.size());

    kharin_m_number_of_sentences_mpi::CountSentencesSequential countSentencesSequential(taskDataSeq);
    ASSERT_EQ(countSentencesSequential.validation(), true);
    countSentencesSequential.pre_processing();
    countSentencesSequential.run();
    countSentencesSequential.post_processing();

    ASSERT_EQ(reference_count[0], 100);
    ASSERT_EQ(reference_count[0], sentence_count[0]);
  }
}

TEST(Parallel_Sentences_Count_MPI, Test_Sentences_with_other_symbols) {
  boost::mpi::communicator world;
  std::string input_text;
  std::vector<int> sentence_count(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  input_text = "Hi! What's you're name? My name is Matthew. How are you? I'm fine, thank you. And you? I'm also fine.";

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input_text.c_str())));
  taskDataPar->inputs_count.emplace_back(input_text.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(sentence_count.data()));
  taskDataPar->outputs_count.emplace_back(sentence_count.size());

  kharin_m_number_of_sentences_mpi::CountSentencesParallel countSentencesParallel(taskDataPar);
  ASSERT_EQ(countSentencesParallel.validation(), true);
  countSentencesParallel.pre_processing();
  countSentencesParallel.run();
  countSentencesParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_count(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input_text.c_str())));
    taskDataSeq->inputs_count.emplace_back(input_text.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_count.data()));
    taskDataSeq->outputs_count.emplace_back(reference_count.size());

    // Run sequential version
    kharin_m_number_of_sentences_mpi::CountSentencesSequential countSentencesSequential(taskDataSeq);
    ASSERT_EQ(countSentencesSequential.validation(), true);
    countSentencesSequential.pre_processing();
    countSentencesSequential.run();
    countSentencesSequential.post_processing();

    // Compare results
    ASSERT_EQ(reference_count[0], 7);
    ASSERT_EQ(reference_count[0], sentence_count[0]);
  }
}

TEST(Parallel_Sentences_Count_MPI, Test_Random_Text) {
  boost::mpi::communicator world;
  std::vector<int> sentence_count(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  // Генерация случайной строки
  char* c_input_text = nullptr;
  int text_length = 0;
  if (world.rank() == 0) {
    std::default_random_engine generator;
    std::uniform_int_distribution<int> length_distribution(10, 100);
    std::string allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .?!";
    std::uniform_int_distribution<int> char_distribution(0, allowed_chars.size() - 1);

    // Длина строки
    text_length = length_distribution(generator);
    c_input_text = new char[text_length + 1];

    // Заполнение случайными символами
    for (int i = 0; i < text_length; i++) {
      c_input_text[i] = allowed_chars[char_distribution(generator)];
    }
    c_input_text[text_length] = '\0';
  }

  boost::mpi::broadcast(world, text_length, 0);
  if (world.rank() != 0) {
    c_input_text = new char[text_length + 1];  // Выделение памяти для других процессов
  }
  boost::mpi::broadcast(world, c_input_text, text_length + 1, 0);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(c_input_text));
  taskDataPar->inputs_count.emplace_back(text_length);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(sentence_count.data()));
  taskDataPar->outputs_count.emplace_back(sentence_count.size());

  // Запуск параллельного подсчета
  kharin_m_number_of_sentences_mpi::CountSentencesParallel countSentencesParallel(taskDataPar);
  ASSERT_EQ(countSentencesParallel.validation(), true);
  countSentencesParallel.pre_processing();
  countSentencesParallel.run();
  countSentencesParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_count(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(c_input_text));
    taskDataSeq->inputs_count.emplace_back(text_length);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_count.data()));
    taskDataSeq->outputs_count.emplace_back(reference_count.size());

    // Запуск последовательного подсчета для проверки
    kharin_m_number_of_sentences_mpi::CountSentencesSequential countSentencesSequential(taskDataSeq);
    ASSERT_EQ(countSentencesSequential.validation(), true);
    countSentencesSequential.pre_processing();
    countSentencesSequential.run();
    countSentencesSequential.post_processing();

    ASSERT_EQ(reference_count[0], sentence_count[0]);
  }
  delete[] c_input_text;  // Очистка динамической памяти
}