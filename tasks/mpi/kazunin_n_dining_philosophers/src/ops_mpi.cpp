#include "mpi/kazunin_n_dining_philosophers/include/ops_mpi.hpp"

#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>

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
