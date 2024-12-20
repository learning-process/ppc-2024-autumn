#include <gtest/gtest.h>

#include <boost/mpi.hpp>
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

namespace mezhuev_m_sobel_edge_detection {

class SobelEdgeDetection : public ppc::core::Task {
 public:
  SobelEdgeDetection(boost::mpi::communicator world_, std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)), world(world_) {}

  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

  const std::vector<int>& get_gradient_x() const { return gradient_x; }
  const std::vector<int>& get_gradient_y() const { return gradient_y; }

 private:
  boost::mpi::communicator world;
  std::vector<int> gradient_x;
  std::vector<int> gradient_y;
};

}  // namespace mezhuev_m_sobel_edge_detection