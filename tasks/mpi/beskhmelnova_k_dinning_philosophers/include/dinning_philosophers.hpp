#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/serialization/serialization.hpp>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace beskhmelnova_k_dining_philosophers {
enum State : std::uint8_t { THINKING, HUNGRY, EATING };

template <typename DataType>
class DiningPhilosophersMPI : public ppc::core::Task {
 public:
  explicit DiningPhilosophersMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  bool check_deadlock() noexcept;

  void think();
  void eat();
  void request_forks();
  void release_forks();

  State state;

 private:
  int num_philosophers;
  bool check_for_termination();
  boost::mpi::communicator world;
  int left_neighbor, right_neighbor;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution;
};
}  // namespace beskhmelnova_k_dining_philosophers