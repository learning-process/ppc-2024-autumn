
#include "mpi/laganina_e_readers_writers/include/ops_mpi.hpp"

#include <chrono>
#include <thread>
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
            (!taskData->outputs_count.empty() && taskData->outputs_count[0] != 0)) &&
           (taskData->outputs.size() > 1);
  }
  return true;
}

bool laganina_e_readers_writers_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  int rank = world.rank();

  count_of_writers = 0;

  int role = 0;
  if (rank != 0) {
    std::random_device dev;
    std::mt19937 gen(dev());
    role = gen() % 2;
  }

  // role == 0 - reader
  // role == 1 - writer

  int size = world.size();

  if (size < 2) {
    return true;
  }

  int work_proc = size - 1;  // flag for terminate program
  int db_w = 1;              // semaphore for locking database for writers
  int readers_count = 0;     // count of readers

  if (rank == 0) {
    while (true) {
      boost::mpi::request req;
      int id_msg;

      // 0 - write
      // 1 - read start
      // 2 - read stop
      // 3 - wait
      // 4 - ready
      // 5 - done
      // 6 - terminate

      req = world.irecv(boost::mpi::any_source, 0, id_msg);
      boost::mpi::status message = req.wait();
      int id_proc = message.source();  // get the process id that sends the message to 0 process

      boost::mpi::request reqs;
      if (id_msg == 0) {
        if (db_w == 1) {
          count_of_writers++;
          int vec_size = static_cast<int>(shared_data.size());
          reqs = world.isend(id_proc, 1, 4);
          reqs = world.isend(id_proc, 2, vec_size);
          reqs.wait();
          reqs = world.isend(id_proc, 2, shared_data.data(), vec_size);
          reqs.wait();
          std::vector<int> new_data(shared_data.size());
          reqs = world.irecv(id_proc, 2, vec_size);
          reqs.wait();  // waiting for getting size of data
          reqs = world.irecv(id_proc, 2, new_data.data(), vec_size);
          reqs.wait();  // waiting for getting data
          shared_data = new_data;
          reqs = world.isend(id_proc, 1, 5);
        } else {
          reqs = world.isend(id_proc, 1, 3);
          reqs.wait();
          continue;
        }
      } else if (id_msg == 1) {
        readers_count++;
        if (db_w == 1) {
          db_w = 0;  // block database for writers
        }
        int vec_size = static_cast<int>(shared_data.size());
        reqs = world.isend(id_proc, 1, 4);
        reqs = world.isend(id_proc, 2, vec_size);
        reqs.wait();
        reqs = world.isend(id_proc, 2, shared_data.data(), vec_size);
        reqs.wait();
      } else if (id_msg == 2) {
        readers_count--;
        if (readers_count == 0) {
          db_w = 1;  // unlock database for writers
        }
        reqs = world.isend(id_proc, 1, 5);
      } else if (id_msg == 6) {
        work_proc--;
        if (work_proc == 0) {
          break;
        }
      }
    }
    res_ = shared_data;
  } else if ((rank != 0) && (role == 1)) {
    boost::mpi::request reqs;
    int message = 0;
    while (message != 4) {
      reqs = world.isend(0, 0, 0);
      reqs.wait();
      reqs = world.irecv(0, 1, message);
      reqs.wait();
    }
    int vec_size = 0;
    reqs = world.irecv(0, 2, vec_size);
    reqs.wait();
    shared_data.resize(vec_size);
    reqs = world.irecv(0, 2, shared_data.data(), vec_size);
    reqs.wait();
    for (auto& t : shared_data) {
      t++;  // adding 1 to each element
    }
    reqs = world.isend(0, 2, vec_size);
    reqs.wait();
    reqs = world.isend(0, 2, shared_data.data(), vec_size);
    reqs.wait();
    reqs = world.irecv(0, 1, message);
    reqs.wait();
    if (message == 5) {
      reqs = world.isend(0, 0, 6);
    }
  } else if ((rank != 0) && (role == 0)) {
    boost::mpi::request reqs;
    int message = 0;
    reqs = world.isend(0, 0, 1);
    reqs.wait();
    reqs = world.irecv(0, 1, message);
    reqs.wait();
    if (message == 4) {
      int vec_size = 0;
      reqs = world.irecv(0, 2, vec_size);
      reqs.wait();
      shared_data.resize(vec_size);
      reqs = world.irecv(0, 2, shared_data.data(), vec_size);
      reqs.wait();
    }
    std::chrono::milliseconds timespan(3);  // simulate reading
    std::this_thread::sleep_for(timespan);

    reqs = world.isend(0, 0, 2);
    reqs.wait();
    reqs = world.irecv(0, 1, message);
    reqs.wait();
    if (message == 5) {
      reqs = world.isend(0, 0, 6);
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
    auto* count = reinterpret_cast<int*>(taskData->outputs[1]);
    *count = count_of_writers;
  }
  return true;
}