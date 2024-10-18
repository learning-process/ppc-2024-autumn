#include "mpi/kolodkin_g_sentence_count/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

int countSentences(const std::string& text) {
  int count = 0;
  for (unsigned long i = 0; i < text.length(); i++) {
    if ((text[i] == '.' || text[i] == '!' || text[i] == '?') &&
        ((text[i + 1] != '.' && text[i + 1] != '!' && text[i + 1] != '?') || i + 1 == text.length())) {
      count++;
    }
  }
  return count;
}

bool kolodkin_g_sentence_count_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  res = 0;
  return true;
}

bool kolodkin_g_sentence_count_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  bool flag1 = (taskData->inputs_count[0] >= 0 && taskData->outputs_count[0] == 1);
  bool flag2 = false;
  if (typeid(*taskData->inputs[0]).name() == typeid(uint8_t).name()) {
    flag2 = true;
  }
  return (flag1 && flag2);
}

bool kolodkin_g_sentence_count_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  res = countSentences(input_);
  return true;
}

bool kolodkin_g_sentence_count_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool kolodkin_g_sentence_count_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
  }
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + proc * delta, delta);
    }
  }
  if (world.rank() == 0) {
    local_input_ = input_.substr(0, delta);
  } else {
    world.recv(0, 0, local_input_.data(), delta);
  }
  res = 0;
  return true;
}

bool kolodkin_g_sentence_count_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  bool flag1 = (taskData->inputs_count[0] >= 0 && taskData->outputs_count[0] == 1);
  bool flag2 = false;
  if (typeid(*taskData->inputs[0]).name() == typeid(uint8_t).name()) {
    flag2 = true;
  }
  return (flag1 && flag2);
}

bool kolodkin_g_sentence_count_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int localSentenceCount = countSentences(local_input_);
  reduce(world, localSentenceCount, res, boost::mpi::minimum<int>(), 0);
  std::this_thread::sleep_for(20ms);
  return true;
}

bool kolodkin_g_sentence_count_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
