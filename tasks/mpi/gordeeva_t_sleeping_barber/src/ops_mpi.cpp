#include "mpi/gordeeva_t_sleeping_barber/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  barber_busy = 0;
  int k = 2;
  res.resize(k, INT_MIN);

  return true;
}

bool gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs_count.size() < 2) return false;
    if (taskData->inputs_count[0] < 0.0) return false;
    if (taskData->outputs_count.size() != 1) return false;
  }
  return true;
}

bool gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  int cl = world.size() - 1;
  max_waiting_chairs = taskData->inputs_count[0];

  if (world.rank() == 0) {
    while (true) {
      int client_id = -1;

      if (world.iprobe(boost::mpi::any_source, 0)) {
        world.recv(boost::mpi::any_source, 0, client_id);
        std::lock_guard<std::mutex> lock(queue_mutex);

        if (static_cast<int>(waiting_clients.size()) < max_waiting_chairs) {
          waiting_clients.push(client_id);
          world.send(client_id, 1, true);
        } else {
          world.send(client_id, 1, false);
          cl--;
        }
      }

      if (!waiting_clients.empty()) {
        serve_next_client();
        cl--;
      } else {
        sleep();
        if (cl == 0) {
          res[0] = 1;
          break;
        }
      }

      if (waiting_clients.empty() && world.size() == 1 && cl == 0) {
        res[0] = 1;
        break;
      }

      if (!world.iprobe(boost::mpi::any_source, 0)) {
        res[0] = 1;
        break;
      }
    }
  } else {
    int client_id = world.rank();

    world.send(0, 0, client_id);

    bool accepted = false;
    world.recv(0, 1, accepted);

    if (accepted) {
      world.recv(0, 2, client_id);
    }
  }

  return true;
}

bool gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    std::copy(res.begin(), res.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  }
  return true;
}

void gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel::serve_next_client() {
  if (!waiting_clients.empty()) {
    int client_id = waiting_clients.front();
    waiting_clients.pop();

    barber_busy = true;
    std::this_thread::sleep_for(2ms);
    world.send(client_id, 0, client_id);
    barber_busy = false;
  }
}

void gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel::sleep() { std::this_thread::sleep_for(1ms); }

bool gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel::add_client_to_queue(int client_id) {
  std::lock_guard<std::mutex> lock(queue_mutex);
  max_waiting_chairs = taskData->inputs_count[0];

  if (static_cast<int>(waiting_clients.size()) < max_waiting_chairs) {
    waiting_clients.push(client_id);
    return true;
  }
  return false;
}
