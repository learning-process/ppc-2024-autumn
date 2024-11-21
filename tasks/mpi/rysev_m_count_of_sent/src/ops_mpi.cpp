// Copyright 2023 Nesterov Alexander


#include <algorithm>
#include <functional>
#include "mpi/rysev_m_count_of_sent/include/ops_mpi.hpp"
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

int rysev_m_count_of_sent_mpi::CountOfSent(std::string& str, bool is_last) {
  char last_symbol = ' ';
  int count = 0;
  for (char symbol : str) {
    if ((symbol == '.' || symbol == '!' || symbol == '?') && last_symbol != '.' && last_symbol != '!' && last_symbol != '?') count += 1;
    last_symbol = symbol;
  }
  if (str.back() != '.' && str.back() != '!' && str.back() != '?' && !str.empty() && is_last) count += 1;
  return count;
}

bool rysev_m_count_of_sent_mpi::CountOfSentSeq::pre_processing() {
  internal_order_test();
  input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  count = 0;
  return true;
}

bool rysev_m_count_of_sent_mpi::CountOfSentSeq::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool rysev_m_count_of_sent_mpi::CountOfSentSeq::run() {
  internal_order_test();
  count = CountOfSent(input_);
  return true;
}

bool rysev_m_count_of_sent_mpi::CountOfSentSeq::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = count;
  return true;
}

bool rysev_m_count_of_sent_mpi::CountOfSentParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) input_ = std::string(std::move(reinterpret_cast<char*>(taskData->inputs[0])));
  count = 0;
  return true;
}

bool rysev_m_count_of_sent_mpi::CountOfSentParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) return taskData->outputs_count[0] == 1;
  return true;
}

bool rysev_m_count_of_sent_mpi::CountOfSentParallel::run() {
  internal_order_test();
  int d = 0;
  int r = 0;
  int l_count;

  if (world.rank() == 0 && world.size() > 1) {
    d = input_.size() / (world.size() - 1);
    r = input_.size() % (world.size() - 1);
  } 
  else if (world.rank() == 0 && world.size() == 1) r = input_.size();
  broadcast(world, d, 0);
  broadcast(world, r, 0);

  if (world.rank() == 0) {
    for (int proc = 1; proc < (world.size()); proc++) world.send(proc, 0, input_.data() + r + (proc - 1) * d, d);
    l_data = std::string(input_.begin(), input_.begin() + r);
  } 
  else {
    l_data = std::string(d, '*');
    world.recv(0, 0, l_data.data(), d);
  }

  l_count = CountOfSent(l_data, world.rank() == world.size() - 1);
  reduce(world, l_count, count, std::plus(), 0);
  return true;
}

bool rysev_m_count_of_sent_mpi::CountOfSentParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) reinterpret_cast<int*>(taskData->outputs[0])[0] = count;
  return true;
}
