#include "mpi/solovev_a_word_count/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace solovev_a_word_count_mpi {

std::string create_text(int quan_words) {
  std::string res;
  std::string word = "word ";
  for (int i = 0; i < quan_words; i++) {
    res += word;
  }
  return res;
}

int word_count(const std::string& input) {
  std::istringstream iss(input);
  return std::distance(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>());
}

bool solovev_a_word_count_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  res = 0;
  boost::mpi::environment env;
  return true;
}

bool solovev_a_word_count_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool solovev_a_word_count_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  res = word_count(input_);
  return true;
}

bool solovev_a_word_count_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool solovev_a_word_count_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  }
  res = 0;
  return true;
}

bool solovev_a_word_count_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return (taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1);
  }
  return true;
}

bool solovev_a_word_count_mpi::TestMPITaskParallel::run() { 
  internal_order_test();
  int l_size;
  int l_count;
  if (input_.size() % world.size() == 0)
    l_size = input_.size() / world.size();
  else
    l_size = input_.size() / world.size() + 1;
  int first = world.rank() * l_size;
  if (first < input_.size()) {
    if (first + l_size >= input_.size()) {
      l_size = input_.size() - first;
    }
    std::string l_str = input_.substr(first, l_size);
    l_count = word_count(l_str);
  } else {
    l_count = 0;
  }
  boost::mpi::reduce(world, l_count, res, std::plus<int>(), 0);
  return true;
}

bool solovev_a_word_count_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}

}  // namespace solovev_a_word_count_mpi
