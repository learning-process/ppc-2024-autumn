#include "mpi/lopatin_i_count_words/include/countWordsMPIHeader.hpp"

namespace lopatin_i_count_words_mpi {

int countWords(const std::string& str) {
  std::istringstream iss(str);
  return std::distance(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>());
}

int divideWords(const std::vector<std::string>& words, int rank, int size) {
  int total_words = words.size();
  int local_words_count = total_words / size;
  int remainder = total_words % size;

  int start = rank * local_words_count + std::min(rank, remainder);
  int end = start + local_words_count + (rank < remainder ? 1 : 0);

  if (start >= total_words) {
    return 0;
  }

  if (end > total_words) {
    end = total_words;
  }

  return end - start;
}

std::string generateLongString(int n) {
  std::string testData;
  std::string testString = "This is a long sentence for performance testing of the word count algorithm using MPI. ";
  for (int i = 0; i < n; i++) {
    testData += testString;
  }
  return testData;
}

bool TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  word_count = 0;
  return true;
}

bool TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool TestMPITaskSequential::run() {
  internal_order_test();
  word_count = countWords(input_);
  return true;
}

bool TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = word_count;
  return true;
}

bool TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  }
  word_count = 0;
  return true;
}

bool TestMPITaskParallel::validation() {
  internal_order_test();
  return (world.rank() == 0) ? (taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1) : true;
}

bool TestMPITaskParallel::run() {
  internal_order_test();
  int input_length = input_.length();
  boost::mpi::broadcast(world, input_length, 0);

  if (world.rank() != 0) {
    input_.resize(input_length);
  }
  boost::mpi::broadcast(world, input_, 0);

  std::vector<std::string> words;

  std::istringstream iss(input_);
  std::string word;
  while (iss >> word) {
    words.push_back(word);
  }

  int local_word_count = divideWords(words, world.rank(), world.size());

  boost::mpi::reduce(world, local_word_count, word_count, std::plus<>(), 0);

  return true;
}

bool TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = word_count;
  }
  return true;
}

}  // namespace lopatin_i_count_words_mpi
