#pragma once

#include <vector>
#include <queue>
#include <climits>

#include "core/task/include/task.hpp"

namespace laganina_e_dejkstras_a_Seq {

class laganina_e_dejkstras_a_Seq : public ppc::core::Task {
 public:
  explicit laganina_e_dejkstras_a_Seq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  static void dijkstra(int start_vertex, const std::vector<int>& row_ptr, const std::vector<int>& col_ind,
                       const std::vector<int>& data, int v, std::vector<int>& distances);

 private:
  std::vector<int> row_ptr;
  std::vector<int> col_ind;
  std::vector<int> data;
  int v{};  // dimension
  std::vector<int> distances;
};

}  // namespace laganina_e_dejkstras_a_Seq