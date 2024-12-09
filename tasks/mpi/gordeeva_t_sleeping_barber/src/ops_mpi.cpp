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
      return false;
    }
  }
  if (world.size() < 3) {
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
  boost::mpi::request reqs;
  while (true) {
    int client_id = -1;

    reqs = world.irecv(1, 0, client_id);
    reqs.wait();
    if (client_id == -1) {
      result = 0;
      return;
    }

    serve_next_client(client_id);
  }
}

void gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel::dispatcher_logic() {
  //boost::mpi::request reqs;
  std::queue<int> waiting_clients;
  max_waiting_chairs = taskData->inputs_count[0];
  int remaining_clients = world.size() - 2;
  bool barber_busy = false;

  while (true) {
    int client_id = -1;

    if (world.iprobe(boost::mpi::any_source, 0)) {
      world.irecv(boost::mpi::any_source, 0, client_id);
      //reqs.wait();

      if (static_cast<int>(waiting_clients.size()) < max_waiting_chairs) {
        waiting_clients.push(client_id);
        world.isend(client_id, 1, true);
        //reqs.wait();
      } else {
        world.isend(client_id, 1, false);
        //reqs.wait();
      }
    }

    if (!barber_busy && !waiting_clients.empty()) {
      int next_client = waiting_clients.front();
      waiting_clients.pop();
      world.isend(0, 0, next_client);
      //reqs.wait();
      barber_busy = true;
    }

    if (world.iprobe(0, 4)) {
      int barber_signal;
      world.irecv(0, 4, barber_signal);
      //reqs.wait();
      barber_busy = false;
    }

    if (waiting_clients.empty() && remaining_clients == 0 && !barber_busy) {
      world.isend(0, 0, -1);
      //reqs.wait();
      break;
    }

    if (world.iprobe(boost::mpi::any_source, 3)) {
      int done_signal;
      world.irecv(boost::mpi::any_source, 3, done_signal);
      //reqs.wait();
      remaining_clients--;
    }
  }
}

void gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel::client_logic() {
  //boost::mpi::request reqs;
  int client_id = world.rank();
  bool accepted = false;

  world.isend(1, 0, client_id);
  //reqs.wait();

  world.irecv(1, 1, accepted);
  //reqs.wait();

  if (accepted) {
    world.irecv(0, 2, client_id);
    //reqs.wait();

    world.isend(1, 3, client_id);
    //reqs.wait();
  } else {
    world.isend(1, 3, client_id);
    //reqs.wait();
  }
}

void gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel::serve_next_client(int client_id) {
  //boost::mpi::request reqs;
  std::this_thread::sleep_for(20ms);
  world.isend(client_id, 2, client_id);
  //reqs.wait();

  world.isend(1, 4, client_id);
  //reqs.wait();
}