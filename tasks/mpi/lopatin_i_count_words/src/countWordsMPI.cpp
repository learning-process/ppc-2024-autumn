#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <sstream>

#include "mpi/lopatin_i_count_words/include/countWordsMPIHeader.hpp"

namespace lopatin_i_count_words_mpi {

int countWords(const std::string& str) {
  std::istringstream iss(str);
  return std::distance(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>());
}

bool TestMPITaskSequential::pre_processing() {
  internal_order_test();
  boost::mpi::environment env;
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

  int local_length = input_length / world.size();
  int start = world.rank() * local_length;
  int end = (world.rank() == world.size() - 1) ? input_length : start + local_length;
  std::string local_input = input_.substr(start, end - start);

  int local_word_count = countWords(local_input);
  boost::mpi::reduce(world, local_word_count, word_count, std::plus<int>(), 0);

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
