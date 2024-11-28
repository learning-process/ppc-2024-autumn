#pragma once

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "core/task/include/task.hpp"

namespace sadikov_I_gather_seq {
std::shared_ptr<ppc::core::TaskData> CreateTaskData(std::vector<int> &InV, const std::vector<int> &CeV,
                                                    std::vector<int> &OtV);
class SumByRowsTask : public ppc::core::Task {
 private:
  std::vector<int> m_sum;
  std::vector<int> m_matrix;
  size_t m_rowsCount, m_columnsCount = 0;

 public:
  explicit SumByRowsTask(std::shared_ptr<ppc::core::TaskData> taskData);
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;
  void Calculate(size_t size);
};
}  // namespace sadikov_I_gather_seq