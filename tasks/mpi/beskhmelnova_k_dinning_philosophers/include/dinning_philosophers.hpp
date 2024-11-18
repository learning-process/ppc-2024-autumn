#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/serialization/serialization.hpp>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace beskhmelnova_k_dining_philosophers {
enum State { THINKING, HUNGRY, EATING };

template <typename DataType>
class DiningPhilosophersMPI : public ppc::core::Task {
 public:
  explicit DiningPhilosophersMPI(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)), world() {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  bool check_deadlock() noexcept;

 private:
  boost::mpi::communicator world;
  State state;
  int left_neighbor, right_neighbor;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution;

  void think();
  void eat();
  void request_forks();
  void release_forks();

  void resolve_deadlock();
  bool check_for_termination();
};
}  // namespace beskhmelnova_k_dining_philosophers