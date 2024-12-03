
#include "mpi/laganina_e_readers_writers/include/ops_mpi.hpp"

#include <chrono>
#include <ctime>
#include <vector>

bool laganina_e_readers_writers_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    int size_data = taskData->inputs_count[0];
    shared_data = std::vector<int>(size_data);
    auto* in_data = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(in_data, in_data + size_data, shared_data.begin());
  }
  return true;
}

bool laganina_e_readers_writers_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return ((!taskData->inputs.empty() && !taskData->outputs.empty()) &&
            (!taskData->inputs_count.empty() && taskData->inputs_count[0] != 0) &&
            (!taskData->outputs_count.empty() && taskData->outputs_count[0] != 0));
  }
  return true;
}

bool laganina_e_readers_writers_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  int rank = world.rank();

  // rank % 2 == 0 - reader
  // rank % 2 == 1 - writer

  int size = world.size();

  if (size < 2) {
    return true;
  }

  int work_proc = size - 1;  // flag for terminate program
  int db_w = 1;              // semaphore for locking database for writers
  int readers_count = 0;     // count of readers

  if (rank == 0) {
    while (true) {
      boost::mpi::status message;
      int id_msg;

      // 0 - write
      // 1 - read start
      // 2 - read stop
      // 3 - wait
      // 4 - ready
      // 5 - done
      // 6 - terminate

      message = world.recv(boost::mpi::any_source, 0, id_msg);
      int id_proc = message.source();  // get the process id that sends the message to "0" process

      if (id_msg == 0) {
        if (db_w == 1) {
          world.send(id_proc, 1, 4);
          world.send(id_proc, 2, shared_data);
          std::vector<int> new_data(shared_data.size());
          world.recv(id_proc, 2, new_data);
          shared_data = new_data;
          world.send(id_proc, 1, 5);
        } else {
          world.send(id_proc, 1, 3);
          continue;
        }
      } else if (id_msg == 1) {
        readers_count++;
        if (db_w == 1) {
          db_w = 0;  // block database for writers
        }
        world.send(id_proc, 1, 4);
        world.send(id_proc, 2, shared_data);
      } else if (id_msg == 2) {
        readers_count--;
        if (readers_count == 0) {
          db_w = 1;  // unlock database for writers
        }
        world.send(id_proc, 1, 5);
      } else if (id_msg == 6) {
        work_proc--;
        if (work_proc == 0) {
          break;
        }
      }
    }
    res_ = shared_data;
  } else if (rank % 2 == 1) {
    int message = 0;
    while (message != 4) {
      world.send(0, 0, 0);
      world.recv(0, 1, message);
    }

    world.recv(0, 2, shared_data);
    for (auto& t : shared_data) {
      t++;  // adding 1 to each element
    }

    world.send(0, 2, shared_data);
    world.recv(0, 1, message);
    if (message == 5) {
      world.send(0, 0, 6);
    }
  } else if (rank % 2 == 0) {
    int message = 0;
    world.send(0, 0, 1);
    world.recv(0, 1, message);
    if (message == 4) {
      world.recv(0, 2, shared_data);
    }
    std::chrono::milliseconds timespan(3);  // simulate reading
    std::this_thread::sleep_for(timespan);

    world.send(0, 0, 2);
    world.recv(0, 1, message);
    if (message == 5) {
      world.send(0, 0, 6);
    }
  }
  world.barrier();
  return true;
}

bool laganina_e_readers_writers_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* out_data = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(res_.begin(), res_.end(), out_data);
  }
  return true;
}
