// ops_mpi.cpp
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

    std::vector<std::string> words;
    std::istringstream stream(input_);
    std::string word;
    while (stream >> word) {
      words.push_back(word);
    }

    int total_words = words.size();
    int chunk_size = total_words / num_processes;

    for (int i = 1; i < num_processes; ++i) {
      int start_pos = i * chunk_size;
      int end_pos = (i == num_processes - 1) ? total_words : (i + 1) * chunk_size;
      std::vector<std::string> chunk(words.begin() + start_pos, words.begin() + end_pos);
      std::ostringstream oss;
      for (const auto& w : chunk) {
        oss << w << " ";
      }
      world.send(i, 0, oss.str());
    }

    words.assign(words.begin(), words.begin() + chunk_size);
    std::ostringstream oss;
    for (const auto& w : words) {
      oss << w << " ";
    }
    input_ = oss.str();
  } else {
    world.recv(0, 0, input_);
  }

  int local_word_count = 0;
  std::istringstream local_stream(input_);
  std::string local_word;
  while (local_stream >> local_word) {
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
