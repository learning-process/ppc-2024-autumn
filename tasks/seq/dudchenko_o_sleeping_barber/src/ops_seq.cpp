#include "seq/dudchenko_o_sleeping_barber/include/ops_seq.hpp"

#include <chrono>
#include <deque>
#include <iostream>
#include <thread>

using namespace std::chrono_literals;

namespace dudchenko_o_sleeping_barber_seq {

bool TestSleepingBarber::pre_processing() {
  result = -1;

  if (taskData && !taskData->inputs_count.empty()) {
    max_wait = taskData->inputs_count[0];
    return max_wait > 0;
  }

  return false;
}

bool TestSleepingBarber::validation() {
  if (!taskData || taskData->inputs_count.empty() || taskData->inputs_count[0] <= 1) {
    return false;
  }

  return true;
}

bool TestSleepingBarber::run() {
  int total_clients = 10;  // Примерное количество клиентов
  std::deque<int> waiting_clients;
  bool barber_busy = false;

  for (int client = 0; client < total_clients; ++client) {
    if (static_cast<int>(waiting_clients.size()) < max_wait) {
      waiting_clients.push_back(client);
    }

    if (!barber_busy && !waiting_clients.empty()) {
      int client = waiting_clients.front();
      waiting_clients.pop_front();
      next_client(client);
      barber_busy = true;
    }

    if (barber_busy && waiting_clients.empty()) {
      barber_busy = false;
    }
  }

  while (!waiting_clients.empty()) {
    int client = waiting_clients.front();
    waiting_clients.pop_front();
    next_client();
  }

  result = 0;
  return true;
}

bool TestSleepingBarber::post_processing() {
  if (!taskData || taskData->outputs.empty() || taskData->outputs_count[0] != sizeof(int)) {
    return false;
  }

  *reinterpret_cast<int*>(taskData->outputs[0]) = result;
  return true;
}

void TestSleepingBarber::next_client() { std::this_thread::sleep_for(20ms); }

}  // namespace dudchenko_o_sleeping_barber_seq
