#include "mpi/volochaev_s_count_characters_27/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/map.hpp>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::string volochaev_s_count_characters_27_mpi::get_random_string(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());

  std::string vec(sz, ' ');
  for (int i = 0; i < sz; i++) {
    vec[i] += gen() % 256;
  }
  return vec;
}

bool volochaev_s_count_characters_27_mpi::Lab1_27_seq::pre_processing() {
  internal_order_test();
  // Init vectors
  auto tmp1 = reinterpret_cast<std::string*>(taskData->inputs[0])[0];
  auto tmp2 = reinterpret_cast<std::string*>(taskData->inputs[0])[1];

  input_ = std::vector<std::pair<char, char>>(std::min(tmp1.size(), tmp2.size()));

  for (size_t i = 0; i < std::min(tmp1.size(), tmp2.size()); i++) {
    input_[i].first = tmp1[i];
    input_[i].second = tmp2[i];
  }

  // Init value for output
  for_res = abs((int)tmp1.size() - (int)tmp2.size());
  res = 0;
  return true;
}

bool volochaev_s_count_characters_27_mpi::Lab1_27_seq::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1;
}

bool volochaev_s_count_characters_27_mpi::Lab1_27_seq::run() {
  internal_order_test();
  for (auto [x, y] : input_) {
    if (x != y) {
      res += 2;
    }
  }
  return true;
}

bool volochaev_s_count_characters_27_mpi::Lab1_27_seq::post_processing() {
  internal_order_test();
  *reinterpret_cast<int*>(taskData->outputs[0]) = res + for_res;
  return true;
}

bool volochaev_s_count_characters_27_mpi::Lab1_27_mpi::pre_processing() {
  internal_order_test();

  unsigned int delta = 0;
  if (world.rank() == 0) {
    auto tmp1 = reinterpret_cast<std::string*>(taskData->inputs[0])[0];
    auto tmp2 = reinterpret_cast<std::string*>(taskData->inputs[0])[1];
    for_res = abs((int)tmp1.size() - (int)tmp2.size());
    delta = (std::min(tmp1.size(), tmp2.size())) / world.size();
    if (std::min(tmp1.size(), tmp2.size()) % world.size() > 0u) ++delta;
  }

  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    // Init vectors
    auto tmp1 = reinterpret_cast<std::string*>(taskData->inputs[0])[0];
    auto tmp2 = reinterpret_cast<std::string*>(taskData->inputs[0])[1];
    input_ = std::vector<std::pair<char, char>>(world.size() * delta);
    for (size_t i = 0; i < std::min(tmp1.size(), tmp2.size()); i++) {
      input_[i].first = tmp1[i];
      input_[i].second = tmp2[i];
    }
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + proc * delta, delta);
    }
  }

  local_input_ = std::vector<std::pair<char, char>>(delta);
  if (world.rank() == 0) {
    local_input_ = std::vector<std::pair<char, char>>(input_.begin(), input_.begin() + delta);
  } else {
    world.recv(0, 0, local_input_.data(), delta);
  }
  // Init value for output
  res = 0;
  return true;
}

bool volochaev_s_count_characters_27_mpi::Lab1_27_mpi::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool volochaev_s_count_characters_27_mpi::Lab1_27_mpi::run() {
  internal_order_test();
  int res1 = 0;
  for (auto [x, y] : local_input_) {
    if (x != y) {
      res1 += 2;
    }
  }
  reduce(world, res1, res, std::plus(), 0);
  return true;
}

bool volochaev_s_count_characters_27_mpi::Lab1_27_mpi::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<int*>(taskData->outputs[0]) = res + for_res;
  }
  return true;
}
