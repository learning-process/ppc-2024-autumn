// Copyright 2023 Nesterov Alexander

#include "mpi/rysev_m_hypercube/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>

using namespace std::chrono_literals;

int rysev_m_gypercube::GyperCube::Next(int c_node, int _target) {
  int d = c_node ^ _target;
  for (int i = 0; i < log2(world.size()); i++) {
    if ((d & (1 << i)) != 0) return c_node ^ (1 << i);
  }
  return c_node;
}

bool rysev_m_gypercube::GyperCube::pre_processing() {
  internal_order_test();
  sender = *reinterpret_cast<int *>(taskData->inputs[0]);
  target = *reinterpret_cast<int *>(taskData->inputs[1]);
  if (world.rank() == sender) data = *reinterpret_cast<int *>(taskData->inputs[2]);
  path.clear();
  done = false;
  return true;
}

bool rysev_m_gypercube::GyperCube::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    int size = world.size();
    if (size < 2 || (size & (size - 1)) != 0) return false;
    if (*reinterpret_cast<int *>(taskData->inputs[0]) >= size || *reinterpret_cast<int *>(taskData->inputs[0]) < 0)
      return false;
  }
  return true;
}

bool rysev_m_gypercube::GyperCube::run() {
  internal_order_test();
  int size = world.size();
  if (world.rank() == sender) {
    path.push_back(world.rank());
    if (world.rank() != target) {
      int next = Next(world.rank(), target);
      world.send(next, 0, done);
      world.send(next, 0, data);
      world.send(next, 0, path);
      world.recv(target, 0, path);
      world.recv(target, 0, done);
    } else {
      return true;
    }
    for (int i = 0; i < size; i++) {
      if (i != sender && std::find(path.begin(), path.end(), i) == path.end()) {
        world.send(i, 0, done);
      }
    }
  } else {
    world.recv(boost::mpi::any_source, 0, done);
    if (!done) {
      world.recv(boost::mpi::any_source, 0, data);
      world.recv(boost::mpi::any_source, 0, path);
      path.push_back(world.rank());
      if (world.rank() == target) {
        done = true;
        world.send(sender, 0, path);
        world.send(sender, 0, done);
      } else {
        int next = Next(world.rank(), target);
        world.send(next, 0, done);
        world.send(next, 0, data);
        world.send(next, 0, path);
      }
    }
  }
  return true;
}

bool rysev_m_gypercube::GyperCube::post_processing() {
  internal_order_test();
  world.barrier();
  if (world.rank() == target) {
    *reinterpret_cast<int *>(taskData->outputs[0]) = data;
    int *p = reinterpret_cast<int *>(taskData->outputs[1]);
    std::copy(path.begin(), path.end(), p);
  }
  return true;
}