// Copyright 2024 Koshkin Matvey
#include "mpi/koshkin_m_dining_philosophers/include/ops_mpi.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <random>

void koshkin_m_dining_philosophers::testMpiTaskParallel::update_neighbors() {
  left_philisopher = (world.rank() + world.size() - 1) % world.size();
  right_philisopher = (world.rank() + 1) % world.size();
}

bool koshkin_m_dining_philosophers::testMpiTaskParallel::pre_processing() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(1, 3);
  if (world.rank() % 2 == 0) {
    status = distrib(gen);
  } else
    status = distrib(gen);
  return true;
}

bool koshkin_m_dining_philosophers::testMpiTaskParallel::validation() {
  int num_philosophers = taskData->inputs_count[0];
  bool is_valid = world.size() >= 2 && num_philosophers >= 2;
  return is_valid;
}

bool koshkin_m_dining_philosophers::testMpiTaskParallel::run() {
  bool is_terminated = false;
  while (!is_terminated) {
    think();
    request_forks();
    eat();
    release_forks();
    if (check_deadlock()) {
      return false;
    }
    is_terminated = check_for_termination();
  }
  return true;
}

bool koshkin_m_dining_philosophers::testMpiTaskParallel::post_processing() {
  world.barrier();
  while (world.iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG)) {
    int leftover_message;
    world.recv(MPI_ANY_SOURCE, MPI_ANY_TAG, leftover_message);
  }
  return true;
}

bool koshkin_m_dining_philosophers::testMpiTaskParallel::check_deadlock() noexcept {
  int local_state = (status == 2) ? 1 : 0;
  std::vector<int> all_states(world.size(), 0);
  boost::mpi::gather(world, local_state, all_states, 0);
  bool deadlock = true;
  if (world.rank() == 0) {
    for (std::size_t i = 0; i < all_states.size(); ++i) {
      if (all_states[i] == 0) {
        deadlock = false;
        break;
      }
    }
  }
  boost::mpi::broadcast(world, deadlock, 0);
  return deadlock;
}

bool koshkin_m_dining_philosophers::testMpiTaskParallel::check_for_termination() {
  std::vector<int> all_statuss(world.size());
  boost::mpi::all_gather(world, status, all_statuss);
  return std::all_of(all_statuss.begin(), all_statuss.end(), [](int s) { return s == 1; });
}

void koshkin_m_dining_philosophers::testMpiTaskParallel::think() {
  status = 1;
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

void koshkin_m_dining_philosophers::testMpiTaskParallel::eat() {
  status = 3;
  std::this_thread::sleep_for(std::chrono::milliseconds(60));
}

void koshkin_m_dining_philosophers::testMpiTaskParallel::request_forks() {
  status = 2;
  world.isend(left_philisopher, 0, 2);
  world.isend(right_philisopher, 0, 2);
  int left_response = 1;
  int right_response = 2;
  world.irecv(left_philisopher, 0, left_response);
  world.irecv(right_philisopher, 0, right_response);
}

void koshkin_m_dining_philosophers::testMpiTaskParallel::release_forks() {
  status = 1;
  world.isend(left_philisopher, 0, 1);
  world.isend(right_philisopher, 0, 1);
  while (world.iprobe(left_philisopher, 0)) {
    int ack;
    world.irecv(left_philisopher, 0, ack);
  }
  while (world.iprobe(right_philisopher, 0)) {
    int ack;
    world.irecv(right_philisopher, 0, ack);
  }
}
