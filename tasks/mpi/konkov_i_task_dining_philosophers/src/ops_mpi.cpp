#include "mpi/konkov_i_task_dining_philosophers/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

namespace konkov_i_dining_philosophers {

DiningPhilosophers::DiningPhilosophers(int num_philosophers)
    : num_philosophers_(num_philosophers), fork_states_(num_philosophers, 0), philosopher_states_(num_philosophers, 0) {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);
}

bool DiningPhilosophers::validation() { return num_philosophers_ > 1; }

bool DiningPhilosophers::pre_processing() {
  if (rank_ == 0) {
    init_philosophers();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool DiningPhilosophers::run() {
  for (int i = 0; i < num_philosophers_; ++i) {
    if (rank_ == i) {
      philosopher_actions(i);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  return true;
}

bool DiningPhilosophers::post_processing() { return !is_deadlock(); }

bool DiningPhilosophers::check_deadlock() { return is_deadlock(); }

bool DiningPhilosophers::check_all_think() {
  return std::all_of(philosopher_states_.begin(), philosopher_states_.end(), [](int state) { return state == 0; });
}

void DiningPhilosophers::init_philosophers() {
  std::fill(fork_states_.begin(), fork_states_.end(), 0);
  std::fill(philosopher_states_.begin(), philosopher_states_.end(), 0);
}

void DiningPhilosophers::philosopher_actions(int id) {
  int left_fork = id;
  int right_fork = (id + 1) % num_philosophers_;

  // Attempt to pick up forks
  if (fork_states_[left_fork] == 0 && fork_states_[right_fork] == 0) {
    fork_states_[left_fork] = 1;
    fork_states_[right_fork] = 1;
    philosopher_states_[id] = 1;  // Eating
  }

  // Release forks
  fork_states_[left_fork] = 0;
  fork_states_[right_fork] = 0;
  philosopher_states_[id] = 0;  // Thinking
}

bool DiningPhilosophers::is_deadlock() {
  return std::all_of(philosopher_states_.begin(), philosopher_states_.end(), [](int state) { return state == 1; });
}

}  // namespace konkov_i_dining_philosophers
