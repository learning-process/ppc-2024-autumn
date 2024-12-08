#include "mpi/dudchenko_o_sleeping_barber/include/ops_mpi.hpp"

#include <functional>
#include <deque>
#include <string>
#include <thread>

using namespace std::chrono_literals;

namespace dudchenko_o_sleeping_barber_mpi {

bool TestMPISleepingBarber::pre_processing() {
  internal_order_test();
  result = -1;

  if (world.rank() == 0) {
    max_wait = taskData->inputs_count[0];
  }

  return true;
}

bool TestMPISleepingBarber::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData->inputs_count.empty() || taskData->inputs_count[0] <= 0) {
      std::cerr << "[VALIDATION] Invalid number of chairs: " << taskData->inputs_count[0] << std::endl;
      return false;
    }
  }
  if (world.size() < 3) {
    std::cerr << "[VALIDATION] Not enough processes. Need at least 3." << std::endl;
    return false;
  }

  return true;
}

bool TestMPISleepingBarber::run() {
  internal_order_test();

  if (world.rank() == 0) {
    while (true) {
      int client = -1;

      world.recv(1, 0, client);

      if (client == -1) {
        result = 0;
        break;
      }

      next_client(client);
    }
  } else if (world.rank() == 1) {
    std::deque<int> waiting_clients;
    max_wait = taskData->inputs_count[0];
    int remaining_clients = world.size() - 2;
    bool barber_busy = false;

    while (true) {
      int client = -1;

      if (world.iprobe(boost::mpi::any_source, 0)) {
        world.recv(boost::mpi::any_source, 0, client);

        if (static_cast<int>(waiting_clients.size()) < max_wait) {
          waiting_clients.push_back(client);
          world.send(client, 1, true);
        } else {
          world.send(client, 1, false);
        }
      }

      if (!barber_busy && !waiting_clients.empty()) {
        int next_client = waiting_clients.front();
        waiting_clients.pop_front();
        world.send(0, 0, next_client);
        barber_busy = true;
      }

      if (world.iprobe(0, 4)) {
        int barber_signal;
        world.recv(0, 4, barber_signal);
        barber_busy = false;
      }

      if (waiting_clients.empty() && remaining_clients == 0 && !barber_busy) {
        world.send(0, 0, -1);
        break;
      }

      if (world.iprobe(boost::mpi::any_source, 3)) {
        int done_signal;
        world.recv(boost::mpi::any_source, 3, done_signal);
        remaining_clients--;
      }
    }
  } else {
    int client = world.rank();
    bool accepted = false;

    world.send(1, 0, client);

    world.recv(1, 1, accepted);

    if (accepted) {
      world.recv(0, 2, client);
      world.send(1, 3, client);
    } else {
      world.send(1, 3, client);
    }
  }

  return true;
}

bool TestMPISleepingBarber::post_processing() {
  internal_order_test();

  world.barrier();

  if (world.rank() == 0) {
    if (!taskData->outputs.empty() && taskData->outputs_count[0] == sizeof(int)) {
      *reinterpret_cast<int*>(taskData->outputs[0]) = result;
    } else {
      return false;
    }
  }

  return true;
}

void TestMPISleepingBarber::next_client(int client) {
  std::this_thread::sleep_for(20ms);
  world.send(client, 2, client);
  world.send(1, 4, client);
}

}  // namespace dudchenko_o_sleeping_barber_mpi
