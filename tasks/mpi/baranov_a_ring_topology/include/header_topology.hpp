#pragma once
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace baranov_a_ring_topology_mpi {
template <class iotype>
class ring_topology : public ppc::core::Task {
 public:
  explicit ring_topology(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(taskData_) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<iotype> input_;

  std::vector<int> poll_;

  std ::vector<iotype> output_;
  int vec_size_;
  boost::mpi::communicator world;
};
}  // namespace baranov_a_ring_topology_mpi
