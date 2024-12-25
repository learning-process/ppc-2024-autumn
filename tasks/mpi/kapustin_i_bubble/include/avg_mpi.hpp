#pragma once

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kapustin_i_bubble_sort_mpi {

class BubbleSortMPI : public ppc::core::Task {
 public:
  explicit BubbleSortMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  std::vector<int> merge(int partner, std::vector<int>& local_data);
  bool validation() override;

  bool pre_processing() override;

  bool run() override;

  bool post_processing() override;

 private:
  std::vector<int> input_;
  std::vector<int> output_;
  std::vector<int> final_result;
  int size_;
  boost::mpi::communicator world;
};

}  // namespace kapustin_i_bubble_sort_mpi