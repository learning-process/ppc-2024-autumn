#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <memory>
#include <random>
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
        generator(world.rank()),
        distribution(100, 500) {
    left_neighbor = (world.rank() + world.size() - 1) % world.size();
    right_neighbor = (world.rank() + 1) % world.size();
  }

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  bool is_waiting_for_fork(int philosopher_id, const std::vector<PhilosophersState>& states);
  void simulate_thinking();
  void simulate_eating();
  void request_forks_from_neighbors();
  void release_forks_to_neighbors();

  bool detect_deadlock() noexcept;
  void resolve_deadlock();
  bool check_termination();

 private:
  boost::mpi::communicator world;
  int left_neighbor, right_neighbor;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution;
};

}  // namespace kazunin_n_dining_philosophers
