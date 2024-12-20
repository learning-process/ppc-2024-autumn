// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include "mpi/zolotareva_a_count_of_words/include/ops_mpi.hpp"

void form(std::string &&str) {
  boost::mpi::communicator world;
  std::string global_string;
  size_t global_count = 0;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_string = str;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_string.data()));
    taskDataPar->inputs_count.emplace_back(global_string.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&global_count));
    taskDataPar->outputs_count.emplace_back(1);
  }

  zolotareva_a_count_of_words_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    size_t reference_count = 0;
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_string.data()));
    taskDataSeq->inputs_count.emplace_back(global_string.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&reference_count));
    taskDataSeq->outputs_count.emplace_back(1);

    zolotareva_a_count_of_words_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_count, global_count);
  }
}
TEST(zolotareva_a_count_of_words_mpi, Test_Count_Words) { form("1"); }
TEST(zolotareva_a_count_of_words_mpi, Test_Two_Words) { form("Hello World"); }

TEST(zolotareva_a_count_of_words_mpi, Test_Leading_Trailing_Spaces) { form("  Hello World  "); }

TEST(zolotareva_a_count_of_words_mpi, Test_Only_Spaces) { form("      "); }

TEST(zolotareva_a_count_of_words_mpi, Test_Complex_Sentence) { form("Multiple Spaces Between Words"); }

TEST(zolotareva_a_count_of_words_mpi, Test_Numbers_And_Spaces) { form(" 1 2 3 4 5 "); }

TEST(zolotareva_a_count_of_words_mpi, Test_Multiple_Consecutive_Letters) { form("A B C D E F G H I J"); }

TEST(zolotareva_a_count_of_words_mpi, Test_Long_String) {
  form(
      "This is a very long string that contains many words spaces and punctuation marks to ensure that the count works "
      "properly");
}

TEST(zolotareva_a_count_of_words_mpi, Test_Very_Long_String) {
  form(
      "My parents are very good and kind. They are young. My mother's name is Natalia. She is thirty years old. She is "
      "a doctor. My mother is very beautiful. My father's name is Victor. He is thirty-two years old. He is an "
      "engineer. I love my parents very much.");
}
TEST(zolotareva_a_count_of_words_mpi, Test_Very_Very_Long_String) {
  form(
      "Children start school at the age of five, but there is some free nursery-school education before that age. The "
      "state nursery schools are not for all. They are for some families, for example for families with only one "
      "parent. In most areas there are private nursery schools. Parents who want their children to go to nursery "
      "school pay for their children under 5 years old to go to these private nursery schools.Some parents prefer "
      "private education. In England and Wales, private schools are called public schools. They are very expensive. "
      "Only 5 per cent of the school population goes to public schools. Public schools are for pupils from 5 or 7 to "
      "18 years old. Some public schools are day schools, but many public schools are boarding schools. Pupils live in "
      "the school and go home in the holidays.");
}