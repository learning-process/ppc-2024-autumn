// Golovkin Maksim Task3
#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/timer.hpp>
#include <cmath>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"

namespace golovkin_linear_image_filtering_with_block_partitioning {

class SimpleBlockMPI : public ppc::core::Task {
 public:
  explicit SimpleBlockMPI(const std::shared_ptr<ppc::core::TaskData>& taskData);

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  const std::vector<int>& getDataPath() const;

 private:
  void distributeData();
  void gatherData();
  void applyGaussianFilter();
  void exchangeHalo();

  boost::mpi::communicator world;

  std::vector<int> original_data_;
  std::vector<int> local_data_;
  std::vector<int> processed_data_;

  size_t total_size_ = 0;
  int width_ = 0;
  int height_ = 0;

  int start_row_ = 0;
  int local_height_ = 0;

  std::vector<int> data_path_;

  const int kernel_[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};

  int extended_local_height_ = 0;
};

}  // namespace golovkin_linear_image_filtering_with_block_partitioning