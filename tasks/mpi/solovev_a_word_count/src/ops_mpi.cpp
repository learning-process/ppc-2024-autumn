#include "mpi/solovev_a_word_count/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace solovev_a_word_count_mpi {

std::vector<char> create_text(int quan_words) {
  std::vector<char> res;
  std::string word = "word ";
  std::string last = "word.";
  for (int i = 0; i < quan_words-1; i++)
    for (unsigned long int symbol = 0; symbol < word.length(); symbol++) 
      {
      res.push_back(word[symbol]);
  }
  for (unsigned long int symbol = 0; symbol < last.length(); symbol++) {
    res.push_back(last[symbol]);
  }
  
  return res;
}


bool solovev_a_word_count_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<char>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  res = 0;
  return true;
}


bool solovev_a_word_count_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool solovev_a_word_count_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  
  for (char symbol : input_) {
    if (symbol != ' ' && symbol != '.') {
    } else {
      res++;
    }
  }
  
  return true;
}
bool solovev_a_word_count_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool solovev_a_word_count_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
  }
  boost::mpi::broadcast(world, delta, 0);
  if (world.rank() == 0) {
    input_ = std ::vector<char>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
    for (unsigned long int i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
    for (int p = 1; p < world.size(); p++) {
      world.send(p, 0, input_.data() + p * delta, delta);
    }
  }
  l_input_.resize(delta);
  if (world.rank() == 0) {
    l_input_ = std::vector<char>(input_.begin(), input_.begin() + delta);
  } else {
    world.recv(0, 0, l_input_.data(), delta);
  }
  res = 0;
  l_res = 0;
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
  for (char symbol : input_) {
    if (symbol != ' ' && symbol != '.') {
    } else {
      l_res++;
    }
  }
  boost::mpi::reduce(world, l_res, res, std::plus<>(), 0);
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
