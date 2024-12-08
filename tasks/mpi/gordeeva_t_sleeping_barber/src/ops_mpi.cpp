#include "mpi/gordeeva_t_sleeping_barber/include/ops_mpi.hpp"

#include <functional>
#include <queue>
#include <string>
#include <thread>

using namespace std::chrono_literals;

bool gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  result = -1;

  if (world.rank() == 0) {
    max_waiting_chairs = taskData->inputs_count[0];
  }

  return true;
}

bool gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel::validation() {
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

bool gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  if (world.rank() == 0) {
    barber_logic();
  } else if (world.rank() == 1) {
    dispatcher_logic();
  } else {
    client_logic();
  }

  return true;
}

bool gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel::post_processing() {
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

void gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel::barber_logic() {
  while (true) {
    int client_id = -1;

    world.recv(1, 0, client_id);

    if (client_id == -1) {
      std::cout << "[BARBER] All clients served. Stopping." << std::endl;
      result = 0;
      return;
    }

    serve_next_client(client_id);
  }
}

void gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel::dispatcher_logic() {
  std::queue<int> waiting_clients;
  max_waiting_chairs = taskData->inputs_count[0];
  int remaining_clients = world.size() - 2;
  bool barber_busy = false;

  while (true) {
    int client_id = -1;

    if (world.iprobe(boost::mpi::any_source, 0)) {
      world.recv(boost::mpi::any_source, 0, client_id);

      if (static_cast<int>(waiting_clients.size()) < max_waiting_chairs) {
        waiting_clients.push(client_id);
        world.send(client_id, 1, true);
        std::cout << "[DISPATCHER] Client " << client_id << " added to the queue. "
                  << "Queue size: " << waiting_clients.size() << "/" << max_waiting_chairs << std::endl;
      } else {
        world.send(client_id, 1, false);
        std::cout << "[DISPATCHER] Client " << client_id << " rejected. Queue is full. "
                  << "Queue size: " << waiting_clients.size() << "/" << max_waiting_chairs << std::endl;
      }
    }

    if (!barber_busy && !waiting_clients.empty()) {
      int next_client = waiting_clients.front();
      waiting_clients.pop();
      world.send(0, 0, next_client);
      barber_busy = true;
    }

    if (world.iprobe(0, 4)) {
      int barber_signal;
      world.recv(0, 4, barber_signal);
      barber_busy = false;
      std::cout << "[DISPATCHER] Barber is now free after serving client " << barber_signal << "." << std::endl;
    }

    if (waiting_clients.empty() && remaining_clients == 0 && !barber_busy) {
      world.send(0, 0, -1);
      std::cout << "[DISPATCHER] All clients served. Sending stop signal to barber." << std::endl;
      break;
    }

    if (world.iprobe(boost::mpi::any_source, 3)) {
      int done_signal;
      world.recv(boost::mpi::any_source, 3, done_signal);
      remaining_clients--;
    }
  }
}

void gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel::client_logic() {
  int client_id = world.rank();
  bool accepted = false;

  world.send(1, 0, client_id);

  world.recv(1, 1, accepted);

  if (accepted) {
    world.recv(0, 2, client_id);
    world.send(1, 3, client_id);
    std::cout << "[CLIENT " << client_id << "] Finished." << std::endl;
  } else {
    world.send(1, 3, client_id);
    std::cout << "[CLIENT " << client_id << "] Queue is full. Leaving." << std::endl;
  }
}

void gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel::serve_next_client(int client_id) {
  std::cout << "[BARBER] Serving client " << client_id << std::endl;
  std::this_thread::sleep_for(20ms);
  world.send(client_id, 2, client_id);
  std::cout << "[BARBER] Finished client " << client_id << std::endl;
  world.send(1, 4, client_id);
}