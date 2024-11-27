#include "mpi/kazunin_n_dining_philosophers/include/ops_mpi.hpp"

#include <iostream>
#include <thread>

namespace kazunin_n_dining_philosophers {
template <typename DataType>
bool KazuninDiningPhilosophersMPI<DataType>::pre_processing() {
  if (world.size() < 2) {
    if (world.rank() == 0) {
      std::cerr << "Error: At least 2 philosophers are required. \n";
    }
    return false;
  }
  return true;
}

template <typename DataType>
bool KazuninDiningPhilosophersMPI<DataType>::validation() {
  return taskData->inputs_count.size() == 1 && taskData->inputs_count[0] >= 2;
}

template <typename DataType>
bool KazuninDiningPhilosophersMPI<DataType>::run() {
  while (!check_termination()) {
    simulate_thinking();
    request_forks_from_neighbors();
    simulate_eating();
    release_forks_to_neighbors();

    if (detect_deadlock()) {
      resolve_deadlock();
    }
  }
  return true;
}

template <typename DataType>
bool KazuninDiningPhilosophersMPI<DataType>::post_processing() {
  if (world.rank() == 0) {
    std::cout << "Dining finished successfully for all philosophers.\n";
  }
  return true;
}

template <typename DataType>
bool KazuninDiningPhilosophersMPI<DataType>::is_waiting_for_fork(int philosopher_id,
                                                                 const std::vector<PhilosophersState>& states) {
  int left_neighbor = (philosopher_id - 1 + world.size()) % world.size();
  int right_neighbor = (philosopher_id + 1) % world.size();

  return (states[left_neighbor] == PhilosophersState::EATING || states[right_neighbor] == PhilosophersState::EATING);
}

template <typename DataType>
void KazuninDiningPhilosophersMPI<DataType>::simulate_thinking() {
  state = PhilosophersState::THINKING;
  std::this_thread::sleep_for(std::chrono::milliseconds(distribution(generator)));
}

template <typename DataType>
void KazuninDiningPhilosophersMPI<DataType>::simulate_eating() {
  state = PhilosophersState::EATING;
  std::this_thread::sleep_for(std::chrono::milliseconds(distribution(generator)));
}

template <typename DataType>
void KazuninDiningPhilosophersMPI<DataType>::request_forks_from_neighbors() {
  state = PhilosophersState::HUNGRY;

  world.send(left_neighbor, 0, state);
  world.send(right_neighbor, 0, state);

  PhilosophersState left_state, right_state;
  world.recv(left_neighbor, 0, left_state);
  world.recv(right_neighbor, 0, right_state);

  if (left_state == PhilosophersState::EATING || right_state == PhilosophersState::EATING) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

template <typename DataType>
void KazuninDiningPhilosophersMPI<DataType>::release_forks_to_neighbors() {
  state = PhilosophersState::THINKING;

  world.send(left_neighbor, 0, state);
  world.send(right_neighbor, 0, state);
}

template <typename DataType>
bool KazuninDiningPhilosophersMPI<DataType>::detect_deadlock() noexcept {
  std::vector<PhilosophersState> states(world.size(), PhilosophersState::THINKING);
  boost::mpi::all_gather(world, state, states);

  int hungry_count = 0;
  for (const auto& s : states) {
    if (s == PhilosophersState::HUNGRY) {
      hungry_count++;
    }
  }

  if (hungry_count == world.size()) {
    for (int i = 0; i < world.size(); ++i) {
      if (is_waiting_for_fork(i, states)) {
        return true;
      }
    }
  }

  return false;
}

template <typename DataType>
void KazuninDiningPhilosophersMPI<DataType>::resolve_deadlock() {
  if (world.rank() == 0) {
    std::cerr << "Deadlock detected! Forcing one philosopher to think.\n";
  }
  state = PhilosophersState::THINKING;
}

template <typename DataType>
bool KazuninDiningPhilosophersMPI<DataType>::check_termination() {
  static int iteration_count = 0;
  const int max_iterations = 10;

  iteration_count++;

  bool local_termination = (iteration_count >= max_iterations);
  bool global_termination = false;

  boost::mpi::all_reduce(world, local_termination, global_termination, std::logical_and<bool>());

  return global_termination;
}

template class KazuninDiningPhilosophersMPI<int>;

}  // namespace kazunin_n_dining_philosophers
