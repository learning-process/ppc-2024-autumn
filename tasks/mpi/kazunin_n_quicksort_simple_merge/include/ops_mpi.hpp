#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>

#include "core/task/include/task.hpp"

namespace kazunin_n_quicksort_simple_merge_mpi {

void iterative_quicksort(std::vector<int>& data);
void worker_function(boost::mpi::communicator& world, const std::vector<int>& local_data);
std::vector<int> master_function(boost::mpi::communicator& world, const std::vector<int>& local_data,
                                 const std::vector<int>& sizes);
void merge_func(boost::mpi::communicator& world, const std::vector<int>& local_data, const std::vector<int>& sizes,
                std::vector<int>& res);
void iterative_quicksort(std::vector<int>& data);

class QuicksortSimpleMerge : public ppc::core::Task {
 public:
  explicit QuicksortSimpleMerge(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int vector_size;
  std::vector<int> input_vector;
  std::vector<int> local_vector;
  std::vector<int> sizes;
  std::vector<int> displs;
  boost::mpi::communicator world;
};

}  // namespace kazunin_n_quicksort_simple_merge_mpi
