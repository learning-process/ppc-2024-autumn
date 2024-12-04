#ifndef OPS_SEQ_HPP
#define OPS_SEQ_HPP
#include <memory.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"

//
namespace kholin_k_iterative_methods_Seidel_seq {
std::vector<float> gen_matrix_with_diag_pred(size_t num_rows, size_t num_colls);
bool IsDiagPred(std::vector<float> row_coeffs, size_t num_colls, size_t start_index, size_t index);
std::vector<float> gen_vector(size_t sz);
float gen_float_value();

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<float> A;
  std::vector<float> X;
  std::vector<float> X_next;
  std::vector<float> X_prev;
  std::vector<float> X0;
  std::vector<float> B;
  std::vector<float> C;
  float epsilon;
  size_t n_rows;
  size_t n_colls;
  static bool CheckDiagPred(float matrix[], size_t num_rows, size_t num_colls);
  static bool IsQuadro(size_t num_rows, size_t num_colls);
  void iteration_perfomance();
  float d();
  static int rank(std::vector<float> matrix, size_t n, size_t m);
  void AllocateBuffers();
  void method_Seidel();
};

}  // namespace kholin_k_iterative_methods_Seidel_seq
#endif