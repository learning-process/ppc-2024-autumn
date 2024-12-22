// Filatev Vladislav Metod Belmana Forda
#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace filatev_v_metod_belmana_forda_seq {

class MetodBelmanaForda : public ppc::core::Task {
 public:
  explicit MetodBelmanaForda(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {};
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int n;
  int m;
  int start;
  std::vector<int> Adjncy;
  std::vector<int> Xadj;
  std::vector<int> Eweights;
  std::vector<int> d;
};

}  // namespace filatev_v_metod_belmana_forda_seq
