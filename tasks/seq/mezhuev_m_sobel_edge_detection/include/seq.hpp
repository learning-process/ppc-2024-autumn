#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace mezhuev_m_sobel_edge_detection {

class SobelEdgeDetectionSeq : public ppc::core::Task {
 public:
  explicit SobelEdgeDetectionSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool run() override;
  bool validation() override;
  bool post_processing() override;

 private:
  std::vector<int16_t> gradient_x;
  std::vector<int16_t> gradient_y;
};

}  // namespace mezhuev_m_sobel_edge_detection