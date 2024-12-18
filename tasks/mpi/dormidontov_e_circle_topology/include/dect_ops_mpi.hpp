#pragma once
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace dormidontov_e_circle_topology_mpi {
class topology : public ppc::core::Task {
 public:
  explicit topology(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  std::vector<int> marks_;
  // I use output because it don't init by default values
  // and i have some issues with send, i see it
  std ::vector<int> output_;
  int size;
  boost::mpi::communicator world;
};
}  // namespace dormidontov_e_circle_topology_mpi