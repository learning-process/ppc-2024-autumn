#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace mezhuev_m_most_different_neighbor_elements {

class MostDifferentNeighborElements : public ppc::core::Task {
 public:
  MostDifferentNeighborElements(boost::mpi::communicator& world_, std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)), world(world_) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int result_[2];
  boost::mpi::communicator world;
};

}  // namespace mezhuev_m_most_different_neighbor_elements