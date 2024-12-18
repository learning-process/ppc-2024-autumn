#pragma once

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace chastov_v_bubble_sort {

template <class T>
class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(taskData_) {
    data_size = (world.rank() == 0) ? taskData->inputs_count[0] : 0;
  }
  bool bubble_sort();
  bool chunk_merge_sort(int neighbor_rank, std::vector<int>& chunk_sizes);
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<T> master_data;
  std::vector<T> chunk_data;
  size_t data_size = 0;
  boost::mpi::communicator world;
};

}  // namespace chastov_v_bubble_sort