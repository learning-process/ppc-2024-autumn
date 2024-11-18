#include <algorithm>
#include <boost/optional/optional_io.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "mpi/solovyev_d_topology_hypercube/include/header.hpp"

std::vector<int> solovyev_d_topology_hypercube_mpi::intToBin(int number, int padding = 0) {
  std::vector<int> result;
  while (number > 0) {
    result.push_back(number % 2);
    number = number / 2;
  }
  while (result.size() < padding) {
    result.push_back(0);
  }
  std::reverse(result.begin(), result.end());
  return result;
}

int solovyev_d_topology_hypercube_mpi::binToInt(std::vector<int> binary) {
  int result{};
  std::reverse(binary.begin(), binary.end());
  for (int i = 0; i < binary.size(); i++) {
    result += (binary[i] * std::pow(2, i));
  }
  return result;
}

int solovyev_d_topology_hypercube_mpi::calcAddress(int src, int dest, int maxAddressLength) {
  std::vector<int> srcBin = intToBin(src, maxAddressLength);
  std::vector<int> destBin = intToBin(dest, maxAddressLength);
  for (int i = 0; i < destBin.size(); i++) {
    if (srcBin[i] != destBin[i]) {
      srcBin[i] = destBin[i];
      break;
    }
  }
  return binToInt(srcBin);
}

bool solovyev_d_topology_hypercube_mpi::TopologyHypercubeMPI::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (reinterpret_cast<int *>(taskData->inputs[0])[1] >= world.size()) {
      return false;
    }
  }
  return true;
}

bool solovyev_d_topology_hypercube_mpi::TopologyHypercubeMPI::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    data_.path.clear();
    int *tmp_ptr = reinterpret_cast<int *>(taskData->inputs[0]);
    int size = (size_t)taskData->inputs_count[0];
    input_ = std::vector<int>(tmp_ptr, tmp_ptr + size);
    data_.value = input_[0];
    data_.destination = input_[1];
    data_.finish = false;
    maxAddressLength = intToBin(world.size() - 1).size();
  }
  return true;
}

bool solovyev_d_topology_hypercube_mpi::TopologyHypercubeMPI::run() {
  internal_order_test();
  // process 0 taking data and send it to first address, then awaiting response from destination process for cleanup
  if (world.rank() == 0) {
    data_.path.push_back(world.rank());
    if (data_.destination != 0) {
      world.send(calcAddress(world.rank(), data_.destination, maxAddressLength), 0, data_);
      world.recv(boost::mpi::any_source, 0, data_);
    } else {
      data_.finish = true;
    }
    // begin cleanup - closing all recv's that not was used in path
    for (int i = 0; i < world.size(); i++) {
      if (std::find(data_.path.begin(), data_.path.end(), i) == data_.path.end()) {
        world.send(i, 0, data_);
      }
    }
  } else {
    world.recv(boost::mpi::any_source, 0, data_);
    if (data_.finish == false) {
      data_.path.push_back(world.rank());
      if (world.rank() == data_.destination) {
        data_.finish = true;
        world.send(0, 0, data_);
      } else {
        world.send(calcAddress(world.rank(), data_.destination, maxAddressLength), 0, data_);
      }
    }
  }
  return true;
}

bool solovyev_d_topology_hypercube_mpi::TopologyHypercubeMPI::post_processing() {
  internal_order_test();
  world.barrier();
  if (world.rank() == 0) {
    reinterpret_cast<int *>(taskData->outputs[0])[0] = data_.value;
    for (int i = 0; i < data_.path.size(); i++) {
      reinterpret_cast<int *>(taskData->outputs[1])[i] = data_.path[i];
    }
  }
  return true;
}
