// Filateva Elizaveta Number_of_sentences_per_line
#include "mpi/filateva_e_number_sentences_line/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

std::string filateva_e_number_sentences_line_mpi::getRandomLine(int max_count) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::string line = "Hello world. How many words are in this sentence? The task of parallel programming!";
  int count = gen() % max_count;
  for (int i = 0; i < count; i++) {
    line = line + line;
  }
  return line;
}

int filateva_e_number_sentences_line_mpi::countSentences(std::string line){
  int count = 0;
  for (auto i = 0; i < line.size(); ++i){
    if (line[i] == '.' || line[i] == '?' || line[i] == '!'){
      ++count;
    }
  }
  return count;
}

bool filateva_e_number_sentences_line_mpi::NumberSentencesLineSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  line = std::string(std::move(reinterpret_cast<char*>(taskData->inputs[0])));
  num = 0;
  return true;
}

bool filateva_e_number_sentences_line_mpi::NumberSentencesLineSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1;
}

bool filateva_e_number_sentences_line_mpi::NumberSentencesLineSequential::run() {
  internal_order_test();
  num = countSentences(line);
  if (line.size() != 0 && line.back() != '.' && line.back() != '?' && line.back() != '!' ){
    ++num;
  }
  return true;
}

bool filateva_e_number_sentences_line_mpi::NumberSentencesLineSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = num;
  return true;
}

bool filateva_e_number_sentences_line_mpi::NumberSentencesLineParallel::pre_processing() {
  internal_order_test();
  unsigned int delta = 0;
  unsigned int remains = 0;
  if (world.rank() == 0) {
    line = std::string(std::move(reinterpret_cast<char*>(taskData->inputs[0])));
    delta = line.size() / world.size();
    remains = line.size() % world.size();
  }
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, line.data() + proc * delta + remains, delta);
    }
    local_line = std::string(line.begin(), line.begin() + delta + remains);
  } else {
    local_line = std::string(delta, '*');
    world.recv(0, 0, local_line.data(), delta);
  }
  // Init value for output
  num = 0;
  return true;
}

bool filateva_e_number_sentences_line_mpi::NumberSentencesLineParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1;
  }
  return true;
}

bool filateva_e_number_sentences_line_mpi::NumberSentencesLineParallel::run() {
  internal_order_test();
  int local_num = countSentences(local_line);
  if (world.rank() == 0 && line.size() != 0 && line.back() != '.' && line.back() != '?' && line.back() != '!' ){
    ++local_num;
  }
  reduce(world, local_num, num, std::plus(), 0);
  return true;
}

bool filateva_e_number_sentences_line_mpi::NumberSentencesLineParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = num;
  }
  return true;
}
