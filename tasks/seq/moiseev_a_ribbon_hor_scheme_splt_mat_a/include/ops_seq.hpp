#include "core/task/include/task.hpp"

namespace moiseev_a_ribbon_hor_scheme_splt_mat_a_seq {

template <typename DataType>
class MatrixMultiplicationSequential : public ppc::core::Task {
 public:
  explicit MatrixMultiplicationSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(taskData_), taskData(taskData_) {}

  bool pre_processing() override {
    internal_order_test();

    if ((taskData->inputs.size() < 2) || taskData->inputs_count.size() != 3) {
      return false;
    }

    m = taskData->inputs_count[0];
    k = taskData->inputs_count[1];
    n = taskData->inputs_count[2];

    auto tmp_ptr_A = reinterpret_cast<DataType*>(taskData->inputs[0]);
    auto tmp_ptr_B = reinterpret_cast<DataType*>(taskData->inputs[1]);
    A.assign(tmp_ptr_A, tmp_ptr_A + m * k);
    B.assign(tmp_ptr_B, tmp_ptr_B + k * n);

    if (A.size() != m * k || B.size() != k * n) {
      return false;
    }

    C.resize(m * n);
    std::fill(C.begin(), C.end(), 0);
    return true;
  }

  bool validation() override {
    internal_order_test();
    return (taskData->inputs.size() == 2 && taskData->inputs_count.size() == 3 && m * k == n * k);
  }

  bool run() override {
    internal_order_test();

    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
        for (size_t p = 0; p < k; ++p) {
          C[i * n + j] += A[i * k + p] * B[p * n + j];
        }
      }
    }
    return true;
  }

  bool post_processing() override {
    internal_order_test();

    if (!taskData->outputs.empty()) {
      auto output_ptr = reinterpret_cast<DataType*>(taskData->outputs[0]);
      std::copy(C.begin(), C.end(), output_ptr);
      return true;
    }
    return false;
  }

 private:
  std::shared_ptr<ppc::core::TaskData> taskData;
  std::vector<DataType> A;
  std::vector<DataType> B;
  std::vector<DataType> C;
  size_t m = 0;
  size_t k = 0;
  size_t n = 0;
};
}  // namespace moiseev_a_ribbon_hor_scheme_splt_mat_a_seq
