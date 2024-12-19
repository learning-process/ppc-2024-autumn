// Filateva Elizaveta Radix Sort

#include <iostream>
#include <list>
#include <vector>

#include "core/task/include/task.hpp"

namespace filateva_e_radix_sort_seq {

class RadixSort : public ppc::core::Task {
 public:
  explicit RadixSort(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int size;
  std::vector<int> arr;
  std::vector<int> ans;
};

}  // namespace filateva_e_radix_sort_seq