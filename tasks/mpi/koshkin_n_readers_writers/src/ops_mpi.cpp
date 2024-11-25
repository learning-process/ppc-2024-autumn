#include "mpi/koshkin_n_readers_writers/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

std::vector<int> koshkin_n_readers_writers_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

bool koshkin_n_readers_writers_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    shared_resource = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      shared_resource[i] = tmp_ptr[i];
    }
    res = {};
  }
  return true;
}

bool koshkin_n_readers_writers_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return ((!taskData->inputs.empty() && !taskData->outputs.empty()) &&
            (!taskData->inputs_count.empty() && taskData->inputs_count[0] != 0) &&
            (!taskData->outputs_count.empty() && taskData->outputs_count[0] != 0));
  }
  return true;
}

bool koshkin_n_readers_writers_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  /* if (world.size() == 1) {
     throw std::runtime_error("This task requires at least two processes to run");
   }*/

  static int resource = 1;   // Semaphore for resource locking
  static int rmutex = 1;     // Semaphore for protection readcount
  static int readcount = 0;  // Global Reader Counter

  if (world.rank() == 0) {
    // The master process controls access
    bool terminate = false;
    int active_processes = world.size() - 1;

    while (!terminate) {
      boost::mpi::status status;
      std::string request;
      status = world.recv(boost::mpi::any_source, 0, request);
      int sender = status.source();

      if (request == "read_entry") {
        if (rmutex != 0) {
          rmutex = 0;  // Block the entrance section
          readcount++;
          if (readcount == 1) {
            if (resource != 0) {
              resource = 0;  // Blocking a resource for writers
            } else {
              world.send(sender, 1, std::string("wait"));
              continue;
            }
          }
          rmutex = 1;  // Unlock section
          world.send(sender, 1, std::string("proceed"));

          world.send(sender, 3, shared_resource);
        } else {
          world.send(sender, 1, std::string("wait"));
        }
      } else if (request == "read_exit") {
        rmutex = 0;  // Block the exit section
        readcount--;
        if (readcount == 0) {
          resource = 1;  // Unlock resource
        }
        rmutex = 1;
        world.send(sender, 1, std::string("done"));
      } else if (request == "write_entry") {
        // Simulation resource.P()
        if (resource != 0) {
          resource = 0;  // Blocking the resource
          world.send(sender, 1, std::string("proceed"));

          // Send the current resource to the writer
          world.send(sender, 3, shared_resource);

        } else {
          world.send(sender, 1, std::string("wait"));
        }
      } else if (request == "write_exit") {
        std::vector<int> updated_resource(shared_resource.size());
        // We receive an updated resource from the writer
        world.recv(sender, 2, updated_resource);

        shared_resource = updated_resource;

        resource = 1;  // Unlocking a resource
        world.send(sender, 1, std::string("done"));

      } else if (request == "terminate") {
        active_processes--;
        if (active_processes == 0) terminate = true;
      }
    }
    res = shared_resource;
  } else {
    std::string role = (world.rank() % 2 == 0) ? "reader" : "writer";
    if (role == "writer") {
      std::string response;
      do {
        world.send(0, 0, std::string("write_entry"));
        world.recv(0, 1, response);
      } while (response == "wait");

      if (response == "proceed") {
        world.recv(0, 3, shared_resource);

        // Simulate recording
        for (auto& val : shared_resource) val += 100;

        world.send(0, 2, shared_resource);
        world.send(0, 0, std::string("write_exit"));
        world.recv(0, 1, response);
      }
    } else if (role == "reader") {
      world.send(0, 0, std::string("read_entry"));
      std::string response;
      world.recv(0, 1, response);
      if (response == "proceed") {
        // Simulate reading
        world.recv(0, 3, shared_resource);
        /*std::cout << "Reader " << world.rank() << " is reading resource: ";
        for (auto val : shared_resource) std::cout << val << " ";
        std::cout << std::endl;*/
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

        world.send(0, 0, std::string("read_exit"));
        world.recv(0, 1, response);
      }
    }
    world.send(0, 0, std::string("terminate"));
  }
  world.barrier();
  return true;
}

bool koshkin_n_readers_writers_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* output = reinterpret_cast<int*>(taskData->outputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      output[i] = res[i];
    }
  }
  return true;
}