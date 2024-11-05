// Copyright 2023 Konkov Ivan
#include "mpi/konkov_i_count_words/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <sstream>

bool konkov_i_count_words_mpi::CountWordsTaskParallel::pre_processing() {
  internal_order_test();
  word_count_ = 0;
  return true;
}

bool konkov_i_count_words_mpi::CountWordsTaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1 && taskData->inputs[0] != nullptr &&
           taskData->outputs[0] != nullptr;
  }
  return true;
}

bool konkov_i_count_words_mpi::CountWordsTaskParallel::run() {
  internal_order_test();
  int num_processes = world.size();
  int rank = world.rank();

  if (rank == 0) {
    input_ = *reinterpret_cast<std::string*>(taskData->inputs[0]);
  }

  // Разделение строки на части для каждого процесса
  int total_length = input_.size();
  int chunk_size = total_length / num_processes;
  int start_pos = rank * chunk_size;
  int end_pos = (rank == num_processes - 1) ? total_length : (rank + 1) * chunk_size;

  std::string local_input = input_.substr(start_pos, end_pos - start_pos);

  int local_word_count = 0;
  std::istringstream stream(local_input);
  std::string word;
  while (stream >> word) {
    local_word_count++;
  }

  boost::mpi::reduce(world, local_word_count, word_count_, std::plus<>(), 0);

  return true;
}

bool konkov_i_count_words_mpi::CountWordsTaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = word_count_;
  }
  return true;
}