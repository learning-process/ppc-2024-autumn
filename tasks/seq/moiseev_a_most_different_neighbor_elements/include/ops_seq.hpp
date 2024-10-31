#include "core/task/include/task.hpp"

namespace moiseev_a_most_different_neighbor_elements_seq {

template <typename DataType>
class MostDifferentNeighborElementsSequential : public ppc::core::Task {
 public:
  explicit MostDifferentNeighborElementsSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(taskData_), taskData(taskData_) {}

  bool pre_processing() override {
    internal_order_test();

    auto tmp_ptr = reinterpret_cast<DataType*>(taskData->inputs[0]);
    input_.assign(tmp_ptr, tmp_ptr + taskData->inputs_count[0]);

    return true;
  }

  bool validation() override {
    internal_order_test();
    return taskData->outputs_count[0] == 2 && taskData->outputs_count[1] == 2;
  }

  bool run() override {
    internal_order_test();
    if (input_.size() < 2) return false;

    DataType max_diff = 0;
    l_elem_index = 0;
    r_elem_index = 1;

    for (size_t i = 0; i < input_.size() - 1; ++i) {
      DataType diff = std::abs(input_[i] - input_[i + 1]);
      if (diff > max_diff) {
        max_diff = diff;
        l_elem_index = i;
        r_elem_index = i + 1;
      }
    }
    return true;
  }

  bool post_processing() override {
    internal_order_test();
    if (taskData->outputs_count[0] >= 2 && taskData->outputs_count[1] >= 2) {
      reinterpret_cast<DataType*>(taskData->outputs[0])[0] = input_[l_elem_index];
      reinterpret_cast<DataType*>(taskData->outputs[0])[1] = input_[r_elem_index];
      reinterpret_cast<uint64_t*>(taskData->outputs[1])[0] = static_cast<uint64_t>(l_elem_index);
      reinterpret_cast<uint64_t*>(taskData->outputs[1])[1] = static_cast<uint64_t>(r_elem_index);
      return true;
    }
    return false;
  }

 private:
  std::shared_ptr<ppc::core::TaskData> taskData;
  std::vector<DataType> input_;
  size_t l_elem_index;
  size_t r_elem_index;
};

}  // namespace moiseev_a_most_different_neighbor_elements_seq
