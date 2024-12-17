﻿
#include "mpi/Odintsov_M_CountingMismatchedCharactersStr/include/ops_mpi.hpp"

#include <cstdlib>
#include <cstring>
#include <ctime>
#include <thread>

using namespace std::chrono_literals;
using namespace Odintsov_M_CountingMismatchedCharactersStr_mpi;

bool CountingCharacterMPISequential::validation() {
  internal_order_test();

  bool ans_out = (taskData->inputs_count[0] == 2);
  bool ans_in = (taskData->outputs_count[0] == 1);
  return (ans_in) && (ans_out);
}
bool CountingCharacterMPISequential::pre_processing() {
  internal_order_test();

  if (strlen(reinterpret_cast<char *>(taskData->inputs[0])) >= strlen(reinterpret_cast<char *>(taskData->inputs[1]))) {
    input.push_back(reinterpret_cast<char *>(taskData->inputs[0]));
    input.push_back(reinterpret_cast<char *>(taskData->inputs[1]));
  } else {
    input.push_back(reinterpret_cast<char *>(taskData->inputs[1]));
    input.push_back(reinterpret_cast<char *>(taskData->inputs[0]));
  }

  ans = 0;
  return true;
}
bool CountingCharacterMPISequential::run() {
  internal_order_test();
  auto *it1 = input[0];
  auto *it2 = input[1];
  while (*it1 != '\0' && *it2 != '\0') {
    if (*it1 != *it2) {
      ans += 2;
    }
    ++it1;
    ++it2;
  }
  ans += std::strlen(it1) + std::strlen(it2);
  return true;
}
bool CountingCharacterMPISequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int *>(taskData->outputs[0])[0] = ans;
  return true;
}

bool CountingCharacterMPIParallel::validation() {
  internal_order_test();

  if (com.rank() == 0) {
    bool ans_out = (taskData->inputs_count[0] == 2);
    bool ans_in = (taskData->outputs_count[0] == 1);
    return (ans_in) && (ans_out);
  }
  return true;
}

bool CountingCharacterMPIParallel::pre_processing() {
  internal_order_test();
  if (com.rank() == 0) {
    if (strlen(reinterpret_cast<char *>(taskData->inputs[0])) >=
        strlen(reinterpret_cast<char *>(taskData->inputs[1]))) {
      input.push_back(reinterpret_cast<char *>(taskData->inputs[0]));
      input.push_back(reinterpret_cast<char *>(taskData->inputs[1]));
    } else {
      input.push_back(reinterpret_cast<char *>(taskData->inputs[1]));
      input.push_back(reinterpret_cast<char *>(taskData->inputs[0]));
    }

    if (strlen(input[0]) != (strlen(input[1]))) {
      ans = strlen(input[0]) - strlen(input[1]);
      input[0][strlen(input[1])] = '\0';
    } else {
      ans = 0;
    }
  }
  return true;
}
bool CountingCharacterMPIParallel::run() {
  internal_order_test();

  size_t loc_size = 0;
  int loc_res = 0;

  if (com.rank() == 0) {
    loc_size = (strlen(input[0]) + com.size() - 1) / com.size();
  }

  broadcast(com, loc_size, 0);

  if (com.rank() == 0) {
    for (int pr = 1; pr < com.size(); pr++) {
      int send_size = std::min(loc_size, strlen(input[0]) - pr * loc_size);

      com.send(pr, 0, input[0] + pr * loc_size, send_size);
      com.send(pr, 0, input[1] + pr * loc_size, send_size);
    }
    std::string str1(input[0], loc_size);
    std::string str2(input[1], loc_size);
    local_input.push_back(str1);
    local_input.push_back(str2);

  } else {
    std::string str1(loc_size, '0');
    std::string str2(loc_size, '0');

    com.recv(0, 0, str1.data(), loc_size);
    com.recv(0, 0, str2.data(), loc_size);

    local_input.push_back(str1);
    local_input.push_back(str2);
  }
  //printf("[Rang %i] str1 %s str2 %s\n", com.rank(), local_input[0].c_str(), local_input[1].c_str());
  //fflush(stdout);
  size_t size_1 = local_input[0].size();
  //printf("Rang %i size %zu\n", com.rank(), size_1);
  //fflush(stdout);
  auto *it1 = input[0];
  auto *it2 = input[1];
  while (*it1 != '\0' && *it2 != '\0') {
    if (*it1 != *it2) {
      ans += 2;
    }
    ++it1;
    ++it2;
  }
  loc_res += std::strlen(it1) + std::strlen(it2);

  reduce(com, loc_res, ans, std::plus(), 0);
  return true;
}

bool CountingCharacterMPIParallel::post_processing() {
  internal_order_test();
  if (com.rank() == 0) {
    reinterpret_cast<int *>(taskData->outputs[0])[0] = ans;
  }
  return true;
}