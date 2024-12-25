#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>

#include "core/task/include/task.hpp"

namespace yasakova_t_quick_sort_with_simple_merge_mpi {

void quicksort_iterative(std::vector<int>& data);
void mpi_worker_function(boost::mpi::communicator& world, const std::vector<int>& local_data);
std::vector<int> master_function(boost::mpi::communicator& world, const std::vector<int>& local_data,
                                 const std::vector<int>& element_sizes);
void mpi_merge_function(boost::mpi::communicator& world, const std::vector<int>& local_data,
                        const std::vector<int>& element_sizes, std::vector<int>& res);

class SimpleMergeQuicksort : public ppc::core::Task {
 public:
  explicit SimpleMergeQuicksort(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int size_of_vector;
  std::vector<int> original_vector;
  std::vector<int> partitioned_vector;
  std::vector<int> element_sizes;
  std::vector<int> displacement;
  boost::mpi::communicator world;
};

}  // namespace yasakova_t_quick_sort_with_simple_merge_mpi