// Copyright 2023 Konkov Ivan
#include "mpi/konkov_i_count_words/include/ops_mpi.hpp"
#include <boost/mpi/collectives.hpp>
#include <sstream>

bool konkov_i_count_words_mpi::CountWordsTaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input_ = *reinterpret_cast<std::string*>(taskData->inputs[0]);
  }
  boost::mpi::broadcast(world, input_, 0);
  word_count_ = 0;
  return true;
}

bool konkov_i_count_words_mpi::CountWordsTaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1;
  }
  return true;
}

bool konkov_i_count_words_mpi::CountWordsTaskParallel::run() {
  internal_order_test();
  std::istringstream stream(input_);
  std::string word;
  int local_word_count = 0;
  while (stream >> word) {
    local_word_count++;
  }
  boost::mpi::reduce(world, local_word_count, word_count_, std::plus<int>(), 0);
  return true;
}

bool konkov_i_count_words_mpi::CountWordsTaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = word_count_;
  }
  return true;
}