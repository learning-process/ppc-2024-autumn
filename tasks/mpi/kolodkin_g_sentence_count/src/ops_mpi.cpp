// Copyright 2023 Nesterov Alexander
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
  for (int i = 0; i < text.length(); i++) {
    if ((text[i] == '.' || text[i] == '!' || text[i] == '?') &&
        ((text[i + 1] != '.' && text[i + 1] != '!' && text[i + 1] != '?') || i + 1 == text.length())) {
      count++;
    }
  }
  return count;
}

bool kolodkin_g_sentence_count_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init string
  input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  // Init value for output
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
  if (world.rank() == 0) {
    // Init vectors
    input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  }
  // Init value for output
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
  int textSize = input_.length();
  MPI_Bcast(&textSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::string localText;
  int chunkSize = textSize / world.size();
  if (world.rank() < world.size() - 1) {
    localText = input_.substr(world.rank() * chunkSize, chunkSize);
  } else {
    localText = input_.substr(world.rank() * chunkSize);  
  }

  int localSentenceCount = countSentences(localText);

  MPI_Reduce(&localSentenceCount, &res, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

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
