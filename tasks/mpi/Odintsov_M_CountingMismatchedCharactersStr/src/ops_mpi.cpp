
#include "mpi/Odintsov_M_CountingMismatchedCharactersStr/include/ops_mpi.hpp"

#include <cstdlib>
#include <cstring>
#include <ctime>
#include <thread>

using namespace std::chrono_literals;
using namespace Odintsov_M_CountingMismatchedCharactersStr_mpi;

// Последовательная версия
bool CountingCharacterMPISequential::validation() {
  internal_order_test();
  // Проверка на то, что у нас 2 строки на входе и одно число на выходе
  bool ans_out = (taskData->inputs_count[0] == 2);
  bool ans_in = (taskData->outputs_count[0] == 1);
  return (ans_in) && (ans_out);
}
bool CountingCharacterMPISequential::pre_processing() {
  internal_order_test();
  // инициализация инпута
  if (strlen(reinterpret_cast<char *>(taskData->inputs[0])) >= strlen(reinterpret_cast<char *>(taskData->inputs[1]))) {
    input.push_back(reinterpret_cast<char *>(taskData->inputs[0]));
    input.push_back(reinterpret_cast<char *>(taskData->inputs[1]));
  } else {
    input.push_back(reinterpret_cast<char *>(taskData->inputs[1]));
    input.push_back(reinterpret_cast<char *>(taskData->inputs[0]));
  }
  // Инициализация ответа
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
// Параллельная версия
bool CountingCharacterMPIParallel::validation() {
  internal_order_test();
  // Проверка на то, что у нас 2 строки на входе и одно число на выходе
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
    // инициализация инпута
    if (strlen(reinterpret_cast<char *>(taskData->inputs[0])) >=
        strlen(reinterpret_cast<char *>(taskData->inputs[1]))) {
      input.push_back(reinterpret_cast<char *>(taskData->inputs[0]));
      input.push_back(reinterpret_cast<char *>(taskData->inputs[1]));
    } else {
      input.push_back(reinterpret_cast<char *>(taskData->inputs[1]));
      input.push_back(reinterpret_cast<char *>(taskData->inputs[0]));
    }
    // Слчай если строки разной длины
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
  // Пересылка
  size_t loc_size = 0;
  // Инициализация в 0 поток
  if (com.rank() == 0) {
    // Инициализация loc_size;
    loc_size = (strlen(input[0]) + com.size() - 1) /
               com.size();  // Округляем вверх, чтобы при большем количестве потоков loc_size = 1
  }
  broadcast(com, loc_size, 0);
  if (com.rank() == 0) {
    for (int pr = 1; pr < com.size(); pr++) {
      size_t send_size =
          std::min(loc_size, strlen(input[0] - pr * loc_size));  // Ограничиваем размар отправляемых данных
      com.send(pr, 0, input[0] + pr * loc_size, send_size);
      com.send(pr, 0, input[1] + pr * loc_size, send_size);
    }
  }
  if (com.rank() == 0) {
    std::string str1(input[0], loc_size);
    std::string str2(input[1], loc_size);
    local_input.push_back(str1);
    local_input.push_back(str2);
  } else {
    std::string str1('0', loc_size);
    std::string str2('0', loc_size);
    com.recv(0, 0, str1.data(), loc_size);
    com.recv(0, 0, str2.data(), loc_size);
    local_input.push_back(str1);
    local_input.push_back(str2);
  }
  size_t size_1 = local_input[0].size();
  //  Реализация
  int loc_res = 0;
  for (size_t i = 0; i < size_1; i++) {
    if (local_input[0][i] != local_input[1][i]) {
      loc_res += 2;
    }
  }
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