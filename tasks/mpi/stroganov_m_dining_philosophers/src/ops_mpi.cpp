// Copyright 2024 Stroganov Mikhail
#include "mpi/stroganov_m_dining_philosophers/include/ops_mpi.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

const std::vector<bool>& stroganov_m_dining_philosophers::TestMPITaskParallel::get_forks() const { return forks; }

bool stroganov_m_dining_philosophers::TestMPITaskParallel::validation() {
  internal_order_test();

  count_philosophers = world.size();

  if (world.rank() == 0) {
    if (!taskData->inputs.empty() && !taskData->inputs_count.empty() &&
        taskData->inputs_count[0] >= static_cast<int>(sizeof(int))) {
      count_philosophers = *reinterpret_cast<int*>(taskData->inputs[0]);
    } else {
      count_philosophers = world.size();
    }
  }

  if (world.size() > 1) {
    broadcast(world, count_philosophers, 0);
  }

  return count_philosophers > 1;
}

bool stroganov_m_dining_philosophers::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (count_philosophers == 0) {
    return false;
  }

  forks = std::vector<bool>(count_philosophers, false);
  dining_philosophers = 0;
  broadcast(world, count_philosophers, 0);
  return true;
}

void stroganov_m_dining_philosophers::TestMPITaskParallel::think(int philosopher_id) {
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  if (world.rank() == 0) {
  }
}

void stroganov_m_dining_philosophers::TestMPITaskParallel::eat(int philosopher_id) {
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  if (world.rank() == 0) {
  }
}

void stroganov_m_dining_philosophers::TestMPITaskParallel::release_forks(int philosopher_id) {
  std::unique_lock<std::mutex> lock(mutex);

  if (count_philosophers == 0) {
    return;
  }

  forks[philosopher_id] = false;
  forks[(philosopher_id + 1) % count_philosophers] = false;
  dining_philosophers--;

  status.notify_all();
}

bool stroganov_m_dining_philosophers::TestMPITaskParallel::distribution_forks(int philosopher_id) {
  std::unique_lock<std::mutex> lock(mutex);

  if (count_philosophers == 0) {
    return false;
  }

  if (philosopher_id % 2 == 0) {
    if (!status.wait_for(lock, std::chrono::milliseconds(100), [this, philosopher_id] {
          return !forks[philosopher_id] && !forks[(philosopher_id + 1) % count_philosophers];
        })) {
      return false;
    }
    forks[philosopher_id] = true;
    forks[(philosopher_id + 1) % count_philosophers] = true;
  } else {
    if (!status.wait_for(lock, std::chrono::milliseconds(100), [this, philosopher_id] {
          return !forks[(philosopher_id + 1) % count_philosophers] && !forks[philosopher_id];
        })) {
      return false;
    }
    forks[(philosopher_id + 1) % count_philosophers] = true;
    forks[philosopher_id] = true;
  }

  dining_philosophers++;
  return true;
}

bool stroganov_m_dining_philosophers::TestMPITaskParallel::run() {
  internal_order_test();

  int philosopher_id = world.rank();
  for (int i = 0; i < 3; ++i) {
    think(philosopher_id);
    if (distribution_forks(philosopher_id)) {
      eat(philosopher_id);
      release_forks(philosopher_id);
    }
  }
  return true;
}

bool stroganov_m_dining_philosophers::TestMPITaskParallel::post_processing() {
  internal_order_test();
  return true;
}