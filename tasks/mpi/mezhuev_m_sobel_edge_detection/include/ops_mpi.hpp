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

namespace mezhuev_m_sobel_edge_detection {

struct TaskData {
  size_t width;
  size_t height;
  std::vector<uint8_t*> inputs;
  std::vector<uint8_t*> outputs;
  std::vector<size_t> inputs_count;
  std::vector<size_t> outputs_count;
};

class SobelEdgeDetectionMPI {
 public:
  SobelEdgeDetectionMPI(boost::mpi::communicator& world) : world(world) {}

  bool validation();
  bool pre_processing(TaskData* task_data);
  bool run();
  bool post_processing();

 private:
  boost::mpi::communicator& world;
  TaskData* taskData = nullptr;
  std::vector<int> gradient_x;
  std::vector<int> gradient_y;
};

}  // namespace mezhuev_m_sobel_edge_detection