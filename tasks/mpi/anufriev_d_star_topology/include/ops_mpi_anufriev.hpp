#pragma once

#include <boost/mpi.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace anufriev_d_star_topology {

class SimpleIntMPI : public ppc::core::Task {
 public:
  explicit SimpleIntMPI(const std::shared_ptr<ppc::core::TaskData>& taskData);

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  void distributeData();
  void gatherData();

  boost::mpi::communicator world;
  std::vector<int> input_data_;
  std::vector<int> processed_data_;
  size_t total_size_;
};

}  // namespace anufriev_d_star_topology