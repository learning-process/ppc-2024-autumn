#include "mpi/kazunin_n_dining_philosophers/include/ops_mpi.hpp"

#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <chrono>
#include <queue>
#include <random>
#include <thread>

namespace kazunin_n_dining_philosophers_mpi {

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
    std::this_thread::sleep_for(std::chrono::milliseconds(think_time));

    request_forks(id, left_fork, right_fork, N, world);

    int eat_time = eat_dist(rng);
    std::this_thread::sleep_for(std::chrono::milliseconds(eat_time));
    eat_count++;

    release_forks(id, left_fork, right_fork, N, world);
  }

  philosophers_comm.barrier();

  int assigned_fork = N + world.rank();
  world.isend(assigned_fork, TERMINATE_FORK, id);

  return true;
}

inline bool fork_manager(int id, boost::mpi::communicator& world) {
  bool fork_available = true;
  bool terminate = false;
  std::queue<int> waiting_queue;
  int philosopher_id;

  while (!terminate) {
    boost::mpi::status s = world.probe(boost::mpi::any_source, boost::mpi::any_tag);

    if (s.tag() == REQUEST_FORK) {
      handle_fork_request(philosopher_id, fork_available, waiting_queue, world, id);
    } else if (s.tag() == RELEASE_FORK) {
      handle_fork_release(philosopher_id, fork_available, waiting_queue, world, id);
    } else if (s.tag() == TERMINATE_FORK) {
      world.recv(s.source(), TERMINATE_FORK, philosopher_id);
      terminate = true;
    }
  }

  return true;
}

inline void request_forks(int id, int left_fork, int right_fork, int N, boost::mpi::communicator& world) {
  if (id % 2 == 0) {
    world.isend(N + left_fork, REQUEST_FORK, id);
    int left_reply;
    world.recv(N + left_fork, FORK_GRANTED, left_reply);

    world.isend(N + right_fork, REQUEST_FORK, id);
    int right_reply;
    world.recv(N + right_fork, FORK_GRANTED, right_reply);
  } else {
    world.isend(N + right_fork, REQUEST_FORK, id);
    int right_reply;
    world.recv(N + right_fork, FORK_GRANTED, right_reply);

    world.isend(N + left_fork, REQUEST_FORK, id);
    int left_reply;
    world.recv(N + left_fork, FORK_GRANTED, left_reply);
  }
}

inline void release_forks(int id, int left_fork, int right_fork, int N, boost::mpi::communicator& world) {
  if (id % 2 == 0) {
    world.isend(N + left_fork, RELEASE_FORK, id);
    world.isend(N + right_fork, RELEASE_FORK, id);
  } else {
    world.isend(N + right_fork, RELEASE_FORK, id);
    world.isend(N + left_fork, RELEASE_FORK, id);
  }
}

inline void handle_fork_request(int& philosopher_id, bool& fork_available, std::queue<int>& waiting_queue,
                                boost::mpi::communicator& world, int id) {
  world.recv(world.probe(boost::mpi::any_source, REQUEST_FORK).source(), REQUEST_FORK, philosopher_id);
  if (fork_available) {
    fork_available = false;
    world.isend(philosopher_id, FORK_GRANTED, id);
  } else {
    waiting_queue.push(philosopher_id);
  }
}

inline void handle_fork_release(int& philosopher_id, bool& fork_available, std::queue<int>& waiting_queue,
                                boost::mpi::communicator& world, int id) {
  world.recv(world.probe(boost::mpi::any_source, RELEASE_FORK).source(), RELEASE_FORK, philosopher_id);
  if (!waiting_queue.empty()) {
    int next_philosopher = waiting_queue.front();
    waiting_queue.pop();
    world.isend(next_philosopher, FORK_GRANTED, id);
  } else {
    fork_available = true;
  }
}

}  // namespace kazunin_n_dining_philosophers_mpi

bool kazunin_n_dining_philosophers_mpi::DiningPhilosophersParallelMPI::validation() {
  internal_order_test();

  int val_eat_limit = *reinterpret_cast<int*>(taskData->inputs[0]);
  int val_min_think_time = *reinterpret_cast<int*>(taskData->inputs[1]);
  int val_max_think_time = *reinterpret_cast<int*>(taskData->inputs[2]);
  int val_min_eat_time = *reinterpret_cast<int*>(taskData->inputs[3]);
  int val_max_eat_time = *reinterpret_cast<int*>(taskData->inputs[4]);

  return val_eat_limit > 0 && val_min_think_time < val_max_think_time && val_min_eat_time < val_max_eat_time &&
         val_max_think_time < 100 && val_min_think_time > 0 && val_max_eat_time < 100 && val_min_eat_time > 0;
}

bool kazunin_n_dining_philosophers_mpi::DiningPhilosophersParallelMPI::pre_processing() {
  internal_order_test();

  eat_limit = *reinterpret_cast<int*>(taskData->inputs[0]);
  min_think_time = *reinterpret_cast<int*>(taskData->inputs[1]);
  max_think_time = *reinterpret_cast<int*>(taskData->inputs[2]);
  min_eat_time = *reinterpret_cast<int*>(taskData->inputs[3]);
  max_eat_time = *reinterpret_cast<int*>(taskData->inputs[4]);
  N = world.size() / 2;
  color = (world.rank() < N) ? 0 : 1;
  local_comm = world.split(color);

  return true;
}

bool kazunin_n_dining_philosophers_mpi::DiningPhilosophersParallelMPI::run() {
  internal_order_test();

  if (color == 0) {
    philosopher(world.rank(), N, world, local_comm, eat_limit, min_think_time, max_think_time, min_eat_time,
                max_eat_time);
  } else {
    fork_manager(world.rank() - N, world);
  }

  return true;
}

bool kazunin_n_dining_philosophers_mpi::DiningPhilosophersParallelMPI::post_processing() {
  internal_order_test();

  return true;
}
