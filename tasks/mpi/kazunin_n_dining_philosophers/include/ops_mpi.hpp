#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <chrono>
// #include <iostream>
#include <queue>
#include <random>
#include <thread>

#include "core/task/include/task.hpp"

namespace kazunin_n_dining_philosophers_mpi {

enum MessageTag { REQUEST_FORK = 1, RELEASE_FORK = 2, FORK_GRANTED = 3, TERMINATE_FORK = 4 };

inline bool philosopher(int id, int N, boost::mpi::communicator& world, boost::mpi::communicator& philosophers_comm,
                        int eat_limit, int min_think_time, int max_think_time, int min_eat_time, int max_eat_time) {
  std::mt19937 rng(id + std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_int_distribution<int> think_dist(min_think_time, max_think_time);
  std::uniform_int_distribution<int> eat_dist(min_eat_time, max_eat_time);

  int left_fork = id;
  int right_fork = (id + 1) % N;

  int eat_count = 0;

  while (eat_count < eat_limit) {
    int think_time = think_dist(rng);
    // std::cout << "Philosopher " << id << " is thinking for " << think_time << " ms.\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(think_time));

    if (id % 2 == 0) {
      // std::cout << "Philosopher " << id << " requests left fork " << left_fork << ".\n";
      world.send(N + left_fork, REQUEST_FORK, id);
      int left_reply;
      world.recv(N + left_fork, FORK_GRANTED, left_reply);
      // std::cout << "Philosopher " << id << " acquired left fork " << left_fork << ".\n";

      // std::cout << "Philosopher " << id << " requests right fork " << right_fork << ".\n";
      world.send(N + right_fork, REQUEST_FORK, id);
      int right_reply;
      world.recv(N + right_fork, FORK_GRANTED, right_reply);
      // std::cout << "Philosopher " << id << " acquired right fork " << right_fork << ".\n";
    } else {
      // std::cout << "Philosopher " << id << " requests right fork " << right_fork << ".\n";
      world.send(N + right_fork, REQUEST_FORK, id);
      int right_reply;
      world.recv(N + right_fork, FORK_GRANTED, right_reply);
      // std::cout << "Philosopher " << id << " acquired right fork " << right_fork << ".\n";

      // std::cout << "Philosopher " << id << " requests left fork " << left_fork << ".\n";
      world.send(N + left_fork, REQUEST_FORK, id);
      int left_reply;
      world.recv(N + left_fork, FORK_GRANTED, left_reply);
      // std::cout << "Philosopher " << id << " acquired left fork " << left_fork << ".\n";
    }

    int eat_time = eat_dist(rng);
    // std::cout << "Philosopher " << id << " is eating for " << eat_time << " ms.\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(eat_time));
    eat_count++;

    if (id % 2 == 0) {
      // std::cout << "Philosopher " << id << " releases left fork " << left_fork << ".\n";
      world.send(N + left_fork, RELEASE_FORK, id);

      // std::cout << "Philosopher " << id << " releases right fork " << right_fork << ".\n";
      world.send(N + right_fork, RELEASE_FORK, id);
    } else {
      // std::cout << "Philosopher " << id << " releases right fork " << right_fork << ".\n";
      world.send(N + right_fork, RELEASE_FORK, id);

      // std::cout << "Philosopher " << id << " releases left fork " << left_fork << ".\n";
      world.send(N + left_fork, RELEASE_FORK, id);
    }
  }

  // std::cout << "Philosopher " << id << " has finished eating.\n";

  philosophers_comm.barrier();

  int assigned_fork = N + world.rank();
  // std::cout << "Philosopher " << id << " sends termination signal to fork " << assigned_fork << ".\n";
  world.send(assigned_fork, TERMINATE_FORK, id);

  return true;
}

inline bool fork_manager(int id, boost::mpi::communicator& world) {
  bool fork_available = true;
  bool terminate = false;
  std::queue<int> waiting_queue;

  while (!terminate) {
    boost::mpi::status s = world.probe(boost::mpi::any_source, boost::mpi::any_tag);
    int philosopher_id;

    if (s.tag() == REQUEST_FORK) {
      world.recv(s.source(), REQUEST_FORK, philosopher_id);
      if (fork_available) {
        fork_available = false;
        // std::cout << "Fork " << id << " is granted to philosopher " << philosopher_id << ".\n";
        world.send(philosopher_id, FORK_GRANTED, id);
      } else {
        waiting_queue.push(philosopher_id);
        // std::cout << "Fork " << id << " is busy. Philosopher " << philosopher_id << " is waiting.\n";
      }
    } else if (s.tag() == RELEASE_FORK) {
      world.recv(s.source(), RELEASE_FORK, philosopher_id);
      // std::cout << "Fork " << id << " is released by philosopher " << philosopher_id << ".\n";
      if (!waiting_queue.empty()) {
        int next_philosopher = waiting_queue.front();
        waiting_queue.pop();
        // std::cout << "Fork " << id << " is granted to philosopher " << next_philosopher << " from the queue.\n";
        world.send(next_philosopher, FORK_GRANTED, id);
      } else {
        fork_available = true;
      }
    } else if (s.tag() == TERMINATE_FORK) {
      world.recv(s.source(), TERMINATE_FORK, philosopher_id);
      // std::cout << "Fork " << id << " received termination signal from philosopher " << philosopher_id << ".\n";
      terminate = true;
    }
  }

  // std::cout << "Fork " << id << " is terminating.\n";
  return true;
}

class DiningPhilosophersParallelMPI : public ppc::core::Task {
 public:
  explicit DiningPhilosophersParallelMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int eat_limit;
  int min_think_time;
  int max_think_time;
  int min_eat_time;
  int max_eat_time;
  int N;
  int color;
  boost::mpi::communicator local_comm;
  boost::mpi::communicator world;
};

}  // namespace kazunin_n_dining_philosophers_mpi
