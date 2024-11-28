// Copyright 2023 Nesterov Alexander
#include "mpi/kalyakina_a_producers_consumers/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

int kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel::ProducersFunction() {
  std::random_device dev;
  std::mt19937 gen(dev());
  int data = (gen() % (20) + 1);
  std::this_thread::sleep_for(std::chrono::milliseconds(data));
  return data;
}
void kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel::ConsumersFunction(int data) {
  std::this_thread::sleep_for(std::chrono::milliseconds(data));
}

bool kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    buffer_size = reinterpret_cast<unsigned int*>(taskData->inputs[0])[0];
    producers_count = reinterpret_cast<int*>(taskData->inputs[0])[1];
    sources = reinterpret_cast<int*>(taskData->inputs[0])[2];
  }
  return true;
}

bool kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of input and output
    if (world.size() < 2) {
      return false;
    }
    if ((taskData->inputs_count[0] != 3) || (taskData->outputs_count[0] != 1)) {
      return false;
    }
    if (reinterpret_cast<int*>(taskData->inputs[0])[0] < 1) {
      return false;
    }
    if ((reinterpret_cast<int*>(taskData->inputs[0])[1] < 1) ||
        (reinterpret_cast<int*>(taskData->inputs[0])[1] > world.size() - 1)) {
      return false;
    }
    return true;
  }
  return true;
}

bool kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel::run() {
  internal_order_test();
  boost::mpi::broadcast(world, producers_count, 0);
  if (world.rank() == 0) {  // Buffer manager
    std::queue<int> buffer;
    std::queue<int> free_consumers;
    int produce_sources = sources;
    int consum_sources = sources;
    boost::mpi::status stat;
    // Freeing up unnecessary processes
    for (int i = std::max(sources, 1); i < producers_count; i++) {
      world.send(i, 0, false);
    }
    for (int i = producers_count + sources; i < world.size(); i++) {
      world.send(i, 1, 0);
    }

    for (int i = 1; i < std::min(sources, producers_count); i++) {
      world.send(i, 0, true);
      produce_sources--;
    }

    for (int i = producers_count; i < std::min(producers_count + sources, world.size()); i++) {
      free_consumers.push(i);
    }

    while (sources > 0) {
      // Receiving data from producers
      if (world.iprobe(MPI_ANY_SOURCE, 0)) {
        if (buffer.size() < buffer_size) {
          stat = world.probe(MPI_ANY_SOURCE, 0);
          int data;
          world.recv(stat.source(), stat.tag(), data);
          buffer.push(data);
          if (produce_sources > 0) {
            world.send(stat.source(), stat.tag(), true);
            produce_sources--;
          } else {
            world.send(stat.source(), stat.tag(), false);
          }
        }
      }
      if ((buffer.size() < buffer_size) && (produce_sources > 0)) {
        buffer.push(ProducersFunction());
        produce_sources--;
      }
      // Receiving consumer release message
      if (world.iprobe(MPI_ANY_SOURCE, 1)) {
        stat = world.probe(MPI_ANY_SOURCE, 1);
        bool answer;
        world.recv(stat.source(), stat.tag(), answer);
        free_consumers.push(stat.source());
        sources--;
      }
      // Sending data to the consumer
      if ((!free_consumers.empty()) && (!buffer.empty())) {
        world.send(free_consumers.front(), 2, buffer.front());
        free_consumers.pop();
        buffer.pop();
        consum_sources--;
      }
    }
    // Freeing up of consumers
    while (!free_consumers.empty()) {
      world.send(free_consumers.front(), 1, 0);
      free_consumers.pop();
    }
    result = sources + produce_sources + consum_sources + buffer.size();
  } else if (world.rank() < producers_count) {  // Producers
    bool answer;
    int tmp;
    while (true) {
      world.recv(0, 0, answer);
      if (!answer) {
        break;
      }
      tmp = ProducersFunction();
      world.send(0, 0, tmp);
    }
  } else {  // Consumers
    int data;
    boost::mpi::status stat;
    while (true) {
      stat = world.probe(0, MPI_ANY_TAG);
      if (stat.tag() == 2) {
        world.recv(0, 2, data);
        ConsumersFunction(data);
        world.send(0, 1, true);
      } else if (stat.tag() == 1) {
        world.recv(0, 1, data);
        break;
      }
    }
  }
  return true;
}

bool kalyakina_a_producers_consumers_mpi::ProducersConsumersTaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<int*>(taskData->outputs[0]) = result;
  }
  return true;
}
