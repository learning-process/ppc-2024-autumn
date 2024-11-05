#include "mpi/konkov_i_count_words/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <random>
#include <regex>
#include <sstream>

bool konkov_i_count_words_mpi::CountWordsTaskParallel::pre_processing() {
  internal_order_test();
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
  int num_processes = world.size();
  int rank = world.rank();

  std::string local_input;

  if (rank == 0) {
    input_ = *reinterpret_cast<std::string*>(taskData->inputs[0]);
  }

  int total_words = 0;
  if (rank == 0) {
    std::regex word_regex("\\b\\w+\\b");
    auto words_begin = std::sregex_iterator(input_.begin(), input_.end(), word_regex);
    auto words_end = std::sregex_iterator();
    total_words = std::distance(words_begin, words_end);
  }

  boost::mpi::broadcast(world, total_words, 0);

  int chunk_size = total_words / num_processes;

  if (rank == 0) {
    std::regex word_regex("\\b\\w+\\b");
    auto words_begin = std::sregex_iterator(input_.begin(), input_.end(), word_regex);
    auto words_end = std::sregex_iterator();
    int current_pos = 0;
    for (int i = 0; i < num_processes; ++i) {
      std::string chunk;
      for (auto it = words_begin; it != words_end && current_pos < (i + 1) * chunk_size; ++it) {
        if (current_pos >= i * chunk_size) {
          chunk += it->str() + " ";
        }
        current_pos++;
      }
      if (i == 0) {
        local_input = chunk;
      } else {
        world.send(i, 0, chunk);
      }
    }
  } else {
    world.recv(0, 0, local_input);
  }

  int local_word_count = 0;
  std::istringstream local_stream(local_input);
  std::string word;
  while (local_stream >> word) {
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

std::string konkov_i_count_words_mpi::CountWordsTaskParallel::generate_random_string(int num_words, int word_length) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 25);

  std::string word(word_length, 'a');
  std::ostringstream oss;
  for (int i = 0; i < num_words; ++i) {
    for (int j = 0; j < word_length; ++j) {
      word[j] = 'a' + dis(gen);
    }
    oss << word << " ";
  }
  return oss.str();
}

std::string konkov_i_count_words_mpi::generate_large_string(int num_words, int word_length) {
  std::string word(word_length, 'a');
  std::ostringstream oss;
  for (int i = 0; i < num_words; ++i) {
    oss << word << " ";
  }
  return oss.str();
}