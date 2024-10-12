#pragma once
#include <string.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"

namespace baranov_a_num_of_orderly_violations_mpi {
template <class iotype, class cntype>
class num_of_orderly_violations : public ppc::core::Task {
 public:
  explicit num_of_orderly_violations(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(taskData_) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  cntype seq_proc(std::vector<iotype> vec);

 private:
  std::vector<iotype> input_;
  std::vector<iotype> loc_vec_;
  cntype num_;
  boost::mpi::communicator world;
  int my_loc_vec_size;
};
}  // namespace baranov_a_num_of_orderly_violations_mpi
