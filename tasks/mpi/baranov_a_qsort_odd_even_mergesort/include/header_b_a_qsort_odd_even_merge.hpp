#pragma once
#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/serialization/vector.hpp>
#include <climits>
#include <cstring>
#include <random>
#include <stack>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace baranov_a_qsort_odd_even_merge_mpi {
template <class iotype>
class baranov_a_odd_even_merge_sort : public ppc::core::Task {
 public:
  explicit baranov_a_odd_even_merge_sort(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(taskData_) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  std::vector<iotype> q_sort_stack(std::vector<iotype>& vec_);

 private:
  std::vector<iotype> input_;

  void merge(std::vector<iotype>& local_data, std::vector<iotype>& other_data);

  std ::vector<iotype> output_;
  int vec_size_;

  std::vector<iotype> loc_vec_;
  boost::mpi::communicator world;
};
}  // namespace baranov_a_qsort_odd_even_merge_mpi
