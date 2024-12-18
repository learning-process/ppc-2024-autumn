// Copyright 2023 Nesterov Alexander
#include "mpi/makhov_m_num_of_diff_elements_in_two_str/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>

int makhov_m_num_of_diff_elements_in_two_str_mpi::countDiffElem(const std::string &str1_, const std::string &str2_) {
  int count = 0;
  int sizeDiff = std::abs(((int)str1_.size() - (int)str2_.size()));
  size_t minSize = std::min(str1_.size(), str2_.size());
  for (size_t i = 0; i < minSize; i++) {
    if (str1_[i] != str2_[i]) count++;
  }
  return count + sizeDiff;
}

std::string makhov_m_num_of_diff_elements_in_two_str_mpi::getShorterStr(std::string str1_, std::string str2_) {
  if (std::min(str1_.size(), str2_.size()) == str1_.size()) return str1_;
  return str2_;
}

bool makhov_m_num_of_diff_elements_in_two_str_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output and strings size
  return taskData->inputs_count[0] >= 0 && taskData->inputs_count[1] >= 0 && taskData->outputs_count[0] == 1;
}

bool makhov_m_num_of_diff_elements_in_two_str_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  str1 = std::string(reinterpret_cast<char *>(taskData->inputs[0]), taskData->inputs_count[0]);
  str2 = std::string(reinterpret_cast<char *>(taskData->inputs[1]), taskData->inputs_count[1]);
  res = 0;
  return true;
}

bool makhov_m_num_of_diff_elements_in_two_str_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  res = countDiffElem(str1, str2);
  return true;
}

bool makhov_m_num_of_diff_elements_in_two_str_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int *>(taskData->outputs[0])[0] = res;
  return true;
}

bool makhov_m_num_of_diff_elements_in_two_str_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  // Init strings in root
  if (world.rank() == 0) {
    str1 = std::string(reinterpret_cast<char *>(taskData->inputs[0]), taskData->inputs_count[0]);
    str2 = std::string(reinterpret_cast<char *>(taskData->inputs[1]), taskData->inputs_count[1]);
    sizeDiff = std::abs((int)(str2.size() - str1.size()));
    str2 = str2.substr(0, str1.size());

    // Init value for output
    res = 0;
  }
  return true;
}

bool makhov_m_num_of_diff_elements_in_two_str_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output and strings size
    return taskData->inputs_count[0] >= 0 && taskData->inputs_count[1] >= 0 && taskData->outputs_count[0] == 1;
  }
  return true;
}

bool makhov_m_num_of_diff_elements_in_two_str_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  std::string str_comb;
  size_t remLen;
  unsigned int delta = 0;
  if (world.rank() == 0) {
    // Get delta
    if (taskData->inputs_count[0] % world.size() == 0)
      delta = taskData->inputs_count[0] / world.size();
    else
      delta = taskData->inputs_count[0] / world.size() + 1;
  }
  broadcast(world, delta, 0);
  if (world.rank() == 0) {
    // Sending each process a string consisting of delta-long parts of the two original strings
    std::string EmptStr((size_t)(delta * 2), ' ');
    for (int process = 1; process < world.size(); process++) {
      if (((int)str1.size() - process * delta) >= 0)
        remLen = (size_t)((int)str1.size() - process * delta);
      else
        remLen = 0;

      if (remLen >= (size_t)(delta)) {
        str_comb = str1.substr((size_t)(process * delta), (size_t)(delta)) +
                   str2.substr((size_t)(process * delta), (size_t)(delta));
        world.send(process, 0, str_comb.data(), delta * 2);
      } else if (remLen > 0) {
        while (remLen < delta) {
          str1 += ' ';
          str2 += ' ';
          remLen++;
        }
        str_comb = str1.substr((size_t)(process * delta), delta) + str2.substr((size_t)(process * delta), delta);
        world.send(process, 0, str_comb.data(), delta * 2);
      } else
        world.send(process, 0, EmptStr.data(), delta * 2);
    }
  }

  if (world.rank() == 0) {
    str1_local = str1.substr(0, delta);
    str2_local = str2.substr(0, delta);
  } else {
    std::string buffer;
    buffer.resize((size_t)(delta * 2));
    world.recv(0, 0, buffer.data(), delta * 2);
    std::string str_comb_local = std::string(buffer.data(), (size_t)(delta * 2));
    str1_local = str_comb_local.substr(0, (size_t)(delta));
    str2_local = str_comb_local.substr((size_t)(delta), (size_t)(delta));
  }
  int local_res = countDiffElem(str1_local, str2_local);
  reduce(world, local_res, res, std::plus(), 0);
  return true;
}

bool makhov_m_num_of_diff_elements_in_two_str_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int *>(taskData->outputs[0])[0] = res + sizeDiff;
  }
  return true;
}
