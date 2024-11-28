#ifndef OPS_SEQ_HPP
#define OPS_SEQ_HPP
#include <memory.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>

#include "core/task/include/task.hpp"

namespace kholin_k_iterative_methods_Seidel_seq {

bool IsDiagPred(float row_coeffs[], size_t num_colls, size_t start_index, size_t index);
void copyA_(float val[], size_t num_rows, size_t num_colls);
float*& getA_();
void freeA_();
void setA_(float val[], size_t num_rows, size_t num_colls);
bool gen_matrix_with_diag_pred(size_t num_rows, size_t num_colls);
float gen_float_value();

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  ~TestTaskSequential() override;

 private:
  float* A;
  float* X;
  float* X_next;
  float* X_prev;
  float* X0;
  float* B;
  float* C;
  float epsilon;
  size_t n_rows;
  size_t n_colls;
  void SetDefault();
  static bool CheckDiagPred(float matrix[], size_t num_rows, size_t num_colls);
  static bool IsQuadro(size_t num_rows, size_t num_colls);
  static float* gen_vector(size_t sz);
  void iteration_perfomance();
  float d();
  void method_Seidel();
};

}  // namespace kholin_k_iterative_methods_Seidel_seq
#endif