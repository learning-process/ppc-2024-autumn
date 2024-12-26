// Golovkin Maksim Task#3

#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace golovkin_linear_image_filtering_with_block_partitioning {

class SimpleIntSEQ : public ppc::core::Task {
 public:
  explicit SimpleIntSEQ(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  void applyGaussianFilterToBlock(int blockRowStart, int blockColStart, int blockSize);
  void applyGaussianFilter();

  std::vector<int> input_data_;
  std::vector<int> processed_data_;
  int rows;
  int cols;

  const int kernel_[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
  const int block_size_ = 16;
};

}  // namespace golovkin_linear_image_filtering_with_block_partitioning