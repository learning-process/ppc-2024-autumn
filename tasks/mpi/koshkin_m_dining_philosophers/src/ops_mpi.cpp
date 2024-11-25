// Copyright 2024 Lyolya Seledkina
#include "mpi/koshkin_m_dining_philosophers/include/ops_mpi.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool koshkin_m_dining_philosophers::TestMPITaskParallel::validation() {
  internal_order_test();

  num_philosophers = world.size();

  if (world.rank() == 0) {
    if (!taskData->inputs.empty() && taskData->inputs_count.size() > 0 &&
        taskData->inputs_count[0] >= static_cast<int>(sizeof(int))) {
      num_philosophers = *reinterpret_cast<int*>(taskData->inputs[0]);
    } else {
      num_philosophers = world.size();
    }
  }

  broadcast(world, num_philosophers, 0);

  return num_philosophers > 1;
}

bool koshkin_m_dining_philosophers::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (num_philosophers == 0) {
    return false;
  }

  forks = std::vector<bool>(num_philosophers, false);
  eating_philosophers = 0;
  broadcast(world, num_philosophers, 0);
  return true;
}

bool koshkin_m_dining_philosophers::TestMPITaskParallel::run() {
  internal_order_test();

  int philosopher_id = world.rank();
  int cycles_completed = 0;
  for (int i = 0; i < 3; ++i) {
    think(philosopher_id);
    if (request_forks(philosopher_id)) {
      eat(philosopher_id);
      cycles_completed++;
      release_forks(philosopher_id);
    }
  }
  return true;
}

bool koshkin_m_dining_philosophers::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
  }
  return true;
}

void koshkin_m_dining_philosophers::TestMPITaskParallel::think(int philosopher_id) {
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  if (world.rank() == 0) {
  }
}

void koshkin_m_dining_philosophers::TestMPITaskParallel::eat(int philosopher_id) {
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  if (world.rank() == 0) {
  }
}

bool koshkin_m_dining_philosophers::TestMPITaskParallel::request_forks(int philosopher_id) {
  std::unique_lock<std::mutex> lock(mutex);

  if (num_philosophers == 0) {
    return false;
  }

  int right_fork = (philosopher_id + 1) % num_philosophers;

  condition.wait(lock, [this, philosopher_id, right_fork] { return can_eat(philosopher_id); });

  forks[philosopher_id] = true;
  forks[right_fork] = true;
  eating_philosophers++;
  return true;
}

void koshkin_m_dining_philosophers::TestMPITaskParallel::release_forks(int philosopher_id) {
  std::unique_lock<std::mutex> lock(mutex);

  if (num_philosophers == 0) {
    return;
  }

  int right_fork = (philosopher_id + 1) % num_philosophers;

  forks[philosopher_id] = false;
  forks[right_fork] = false;
  eating_philosophers--;

  condition.notify_all();
}

bool koshkin_m_dining_philosophers::TestMPITaskParallel::can_eat(int philosopher_id) {
  if (num_philosophers == 0) {
    return false;
  }

  bool left_fork_free = !forks[philosopher_id];
  int right_fork = (philosopher_id + 1) % num_philosophers;
  bool right_fork_free = !forks[right_fork];

  return left_fork_free && right_fork_free;
}
