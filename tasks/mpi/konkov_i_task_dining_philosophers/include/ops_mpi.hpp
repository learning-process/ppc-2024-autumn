#ifndef MODULES_TASK_2_KONKOV_I_DINING_PHILOSOPHERS_OPS_MPI_HPP_
#define MODULES_TASK_2_KONKOV_I_DINING_PHILOSOPHERS_OPS_MPI_HPP_

#include <mpi.h>

#include <vector>

namespace konkov_i_dining_philosophers {

class DiningPhilosophers {
 public:
  explicit DiningPhilosophers(int num_philosophers);
  bool validation() const;
  bool pre_processing();
  bool run();
  bool post_processing();
  bool check_deadlock();
  bool check_all_think();

 private:
  int num_philosophers_;
  int rank_;
  int size_;
  std::vector<int> fork_states_;
  std::vector<int> philosopher_states_;

  void init_philosophers();
  void philosopher_actions(int id);
  bool is_deadlock();
};

}  // namespace konkov_i_dining_philosophers

#endif  // MODULES_TASK_2_KONKOV_I_DINING_PHILOSOPHERS_OPS_MPI_HPP_
