#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <queue>

#include "core/task/include/task.hpp"

namespace kazunin_n_dining_philosophers_mpi {

enum MessageTag : std::uint8_t { REQUEST_FORK = 1, RELEASE_FORK = 2, FORK_GRANTED = 3, TERMINATE_FORK = 4 };

inline void request_forks(int id, int left_fork, int right_fork, int N, boost::mpi::communicator& world);
inline void release_forks(int id, int left_fork, int right_fork, int N, boost::mpi::communicator& world);
inline void handle_fork_request(int& philosopher_id, bool& fork_available, std::queue<int>& waiting_queue,
                                boost::mpi::communicator& world, int id);
inline void handle_fork_release(int& philosopher_id, bool& fork_available, std::queue<int>& waiting_queue,
                                boost::mpi::communicator& world, int id);
inline bool fork_manager(int id, boost::mpi::communicator& world);
inline bool philosopher(int id, int N, boost::mpi::communicator& world, boost::mpi::communicator& philosophers_comm,
                        int eat_limit, int min_think_time, int max_think_time, int min_eat_time, int max_eat_time);

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
