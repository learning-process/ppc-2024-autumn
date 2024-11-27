#pragma once

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace kovalev_k_bubble_sort_oddeven_transposition_mpi {
template <class T>
class BubbleSortOddEvenTranspositionPar : public ppc::core::Task {
 private:
  std::vector<T> glob_v;
  std::vector<T> loc_v;
  size_t n = 0;
  boost::mpi::communicator world;

 public:
  explicit BubbleSortOddEvenTranspositionPar(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(taskData_) {
    if (world.rank() == 0) n = taskData->inputs_count[0];
  }
  bool bubble_sort_mpi();
  bool divide_and_merge(int partner, std::vector<size_t>& sendcounts);
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
};
}  // namespace kovalev_k_bubble_sort_oddeven_transposition_mpi