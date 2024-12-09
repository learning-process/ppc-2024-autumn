#include "mpi/shkurinskaya_e_count_sentences/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;
bool shkurinskaya_e_count_sentences_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  text = *reinterpret_cast<std::string*>(taskData->inputs[0]);
  return true;
}
bool shkurinskaya_e_count_sentences_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool shkurinskaya_e_count_sentences_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  bool in_end = false;
  for (size_t i = 0; i < text.size(); i++) {
    char ch = text[i];
    if (ch == '!' || ch == '?' || ch == '.') {
      if (!in_end) {
        res++;
        in_end = true;
      }
    } else if (ch != ' ') {
      in_end = false;
    }
  }
  return true;
}

bool shkurinskaya_e_count_sentences_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool shkurinskaya_e_count_sentences_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    text = *reinterpret_cast<std::string*>(taskData->inputs[0]);
  }
  local_res = 0;
  return true;
}

bool shkurinskaya_e_count_sentences_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool shkurinskaya_e_count_sentences_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  size_t delta;
  size_t remainder;
  if (world.rank() == 0) {
    size_t total_size = text.size();
    delta = total_size / world.size();
    remainder = total_size % world.size();
  }
  broadcast(world, delta, 0);
  broadcast(world, remainder, 0);
  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); ++proc) {
      size_t start_index = proc * delta;
      size_t length = (proc == world.size() - 1) ? delta + remainder : delta;
      world.send(proc, 0, text.data() + start_index, length);
    }
    size_t length = (world.size() == 1) ? text.size() : delta;
    local_input_.assign(text.begin(), text.begin() + length);
  } else {
    size_t length = (world.rank() == world.size() - 1) ? delta + remainder : delta;
    local_input_.resize(length);
    world.recv(0, 0, local_input_.data(), length);
  }
  bool in_end = false;
  if (world.rank() != 0) {
    world.recv(world.rank() - 1, 0, in_end);
  }
  for (size_t i = 0; i < local_input_.size(); ++i) {
    char ch = local_input_[i];
    if (ch == '!' || ch == '?' || ch == '.') {
      if (!in_end) {
        local_res++;
        in_end = true;
      }
    } else if (ch != ' ') {
      in_end = false;
    }
  }
  if (world.rank() != world.size() - 1) {
    world.send(world.rank() + 1, 0, in_end);
  }
  boost::mpi::reduce(world, local_res, res, std::plus<>(), 0);
  return true;
}

bool shkurinskaya_e_count_sentences_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
