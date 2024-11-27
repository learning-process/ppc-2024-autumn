#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/serialization/serialization.hpp>
#include <chrono>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "core/task/include/task.hpp"

namespace kazunin_n_dining_philosophers {
enum class PhilosophersState { THINKING, HUNGRY, EATING };

template <typename DataType>
class KazuninDiningPhilosophersMPI : public ppc::core::Task {
 public:
  PhilosophersState state;

  explicit KazuninDiningPhilosophersMPI(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)),
        state(PhilosophersState::THINKING),
        world(),
        rank(world.rank()),
        left_neighbor((rank + world.size() - 1) % world.size()),
        right_neighbor((rank + 1) % world.size()),
        generator(rank),
        distribution(100, 500) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void simulate_thinking();
  void simulate_eating();
  void request_forks_from_neighbors();
  void release_forks_to_neighbors();
  bool check_deadlock() noexcept;
  void resolve_deadlock();
  bool check_termination();

 private:
  boost::mpi::communicator world;
  int rank;
  int left_neighbor, right_neighbor;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution;
};

template <typename DataType>
bool kazunin_n_dining_philosophers::KazuninDiningPhilosophersMPI<DataType>::pre_processing() {
  state = PhilosophersState::THINKING;
  return true;
}

template <typename DataType>
bool kazunin_n_dining_philosophers::KazuninDiningPhilosophersMPI<DataType>::validation() {
  int count_philosophers = taskData->inputs_count[0];
  return (world.size() >= 2 && count_philosophers >= 2);
}

template <typename DataType>
bool kazunin_n_dining_philosophers::KazuninDiningPhilosophersMPI<DataType>::run() {
  while (true) {
    simulate_thinking();
    request_forks_from_neighbors();
    simulate_eating();
    release_forks_to_neighbors();
    if (check_deadlock()) resolve_deadlock();
    if (check_termination()) break;
  }
  return true;
}

template <typename DataType>
bool kazunin_n_dining_philosophers::KazuninDiningPhilosophersMPI<DataType>::post_processing() {
  world.barrier();
  std::vector<PhilosophersState> all_states(world.size());
  boost::mpi::all_gather(world, state, all_states);
  return true;
}

template <typename DataType>
bool kazunin_n_dining_philosophers::KazuninDiningPhilosophersMPI<DataType>::check_deadlock() noexcept {
  int local_hungry = (state == PhilosophersState::HUNGRY) ? 1 : 0;
  int total_hungry = 0;
  boost::mpi::all_reduce(world, local_hungry, total_hungry, std::plus<int>());
  bool deadlock = (total_hungry == world.size());
  if (deadlock) {
  }
  return deadlock;
}

template <typename DataType>
void kazunin_n_dining_philosophers::KazuninDiningPhilosophersMPI<DataType>::resolve_deadlock() {
  if (rank == 0) {
    world.send(1, 0, PhilosophersState::THINKING);
  }
}

template <typename DataType>
bool kazunin_n_dining_philosophers::KazuninDiningPhilosophersMPI<DataType>::check_termination() {
  int local_thinking = (state == PhilosophersState::THINKING) ? 1 : 0;
  int total_thinking = 0;
  boost::mpi::all_reduce(world, local_thinking, total_thinking, std::plus<int>());
  bool termination = (total_thinking == world.size());
  if (termination) {
  }
  return termination;
}

template <typename DataType>
void kazunin_n_dining_philosophers::KazuninDiningPhilosophersMPI<DataType>::simulate_thinking() {
  state = PhilosophersState::THINKING;
  std::this_thread::sleep_for(std::chrono::milliseconds(distribution(generator)));
}

template <typename DataType>
void kazunin_n_dining_philosophers::KazuninDiningPhilosophersMPI<DataType>::simulate_eating() {
  state = PhilosophersState::EATING;
  std::this_thread::sleep_for(std::chrono::milliseconds(distribution(generator)));
}

template <typename DataType>
void kazunin_n_dining_philosophers::KazuninDiningPhilosophersMPI<DataType>::request_forks_from_neighbors() {
  state = PhilosophersState::HUNGRY;
  world.send(left_neighbor, 0, state);
  world.send(right_neighbor, 0, state);

  PhilosophersState left_response, right_response;
  world.recv(left_neighbor, 0, left_response), world.recv(right_neighbor, 0, right_response);

  if (left_response == PhilosophersState::THINKING && right_response == PhilosophersState::THINKING) {
    state = PhilosophersState::EATING;
  }
}

template <typename DataType>
void kazunin_n_dining_philosophers::KazuninDiningPhilosophersMPI<DataType>::release_forks_to_neighbors() {
  state = PhilosophersState::THINKING;
  world.isend(left_neighbor, 0, state);
  world.isend(right_neighbor, 0, state);
};
}  // namespace kazunin_n_dining_philosophers