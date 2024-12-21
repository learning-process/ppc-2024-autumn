#include <algorithm>
#include <vector>

#include "core/task/include/task.hpp"

namespace mezhuev_m_most_different_neighbor_elements {

class MostDifferentNeighborElements : public ppc::core::Task {
 public:
  explicit MostDifferentNeighborElements(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

  const std::vector<int>& getInput() const { return input; }
  const std::vector<int>& getResult() const { return result; }

 private:
  std::vector<int> input;
  std::vector<int> result;
};

}  // namespace mezhuev_m_most_different_neighbor_elements