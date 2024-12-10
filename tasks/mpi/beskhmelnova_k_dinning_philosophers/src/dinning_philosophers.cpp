#include "mpi/beskhmelnova_k_dinning_philosophers/include/dinning_philosophers.hpp"

#include <chrono>
#include <thread>

template <typename DataType>
bool beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<DataType>::pre_processing() {
  left_neighbor = (world.rank() + world.size() - 1) % world.size();
  right_neighbor = (world.rank() + 1) % world.size();
  state = THINKING;
  generator.seed(world.rank());
  distribution = std::uniform_int_distribution<int>(1, 3);
  return true;
}

template <typename DataType>
bool beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<DataType>::validation() {
  int num_philosophers = taskData->inputs_count[0];
  return world.size() >= 2 && num_philosophers >= 2;
}

template <typename DataType>
bool beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<DataType>::run() {
  while (!check_for_termination()) {
    think();
    request_forks();
    eat();
    release_forks();
    if (check_deadlock()) resolve_deadlock();
  }
  return true;
}

template <typename DataType>
bool beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<DataType>::post_processing() {
  world.barrier();
  while (world.iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG)) {
    State leftover_message;
    world.recv(MPI_ANY_SOURCE, MPI_ANY_TAG, leftover_message);
  }
  return true;
}

template <typename DataType>
bool beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<DataType>::check_deadlock() noexcept {
  int local_state = (state == HUNGRY) ? 1 : 0;
  std::vector<State> all_states(world.size(), THINKING);
  std::vector<int> gathered_states(world.size(), 0);
  boost::mpi::gather(world, local_state, gathered_states, 0);
  if (world.rank() == 0)
    for (std::size_t i = 0; i < gathered_states.size(); ++i) all_states[i] = static_cast<State>(gathered_states[i]);
  bool deadlock = true;
  if (world.rank() == 0) {
    for (std::size_t i = 0; i < all_states.size(); ++i) {
      if (all_states[i] == THINKING) {
        deadlock = false;
        break;
      }
    }
  }
  boost::mpi::broadcast(world, deadlock, 0);
  return deadlock;
}

template <typename DataType>
void beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<DataType>::resolve_deadlock() {
  if (world.rank() == 0) {
    int philosopher_to_release = std::rand() % world.size();
    world.send(philosopher_to_release, 1, THINKING);
  }
  bool resolved = false;
  while (!resolved) {
    if (world.iprobe(0, 1)) {
      State release_signal;
      world.recv(0, 1, release_signal);
      if (release_signal == THINKING) {
        state = THINKING;
        resolved = true;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

template <typename DataType>
bool beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<DataType>::check_for_termination() {
  bool all_thinking = true;
  std::vector<State> all_states(world.size(), THINKING);
  boost::mpi::all_gather(world, state, all_states);
  for (std::size_t i = 0; i < all_states.size(); ++i) {
    if (all_states[i] != THINKING) {
      all_thinking = false;
      break;
    }
  }
  return all_thinking;
}

template <typename DataType>
void beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<DataType>::think() {
  state = THINKING;
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

template <typename DataType>
void beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<DataType>::eat() {
  state = EATING;
  std::this_thread::sleep_for(std::chrono::milliseconds(60));
}

template <typename DataType>
void beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<DataType>::request_forks() {
  state = HUNGRY;
  world.send(left_neighbor, 0, HUNGRY);
  world.send(right_neighbor, 0, HUNGRY);
  State left_response;
  State right_response;
  world.recv(left_neighbor, 0, left_response);
  world.recv(right_neighbor, 0, right_response);
}

template <typename DataType>
void beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<DataType>::release_forks() {
  state = THINKING;
  world.send(left_neighbor, 0, THINKING);
  world.send(right_neighbor, 0, THINKING);
  while (world.iprobe(left_neighbor, 0)) {
    State ack;
    world.recv(left_neighbor, 0, ack);
  }
  while (world.iprobe(right_neighbor, 0)) {
    State ack;
    world.recv(right_neighbor, 0, ack);
  }
}

template class beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<int>;
