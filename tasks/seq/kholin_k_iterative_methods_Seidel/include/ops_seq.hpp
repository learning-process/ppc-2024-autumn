#ifndef OPS_SEQ_HPP
#define OPS_SEQ_HPP
#include <memory.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace kholin_k_iterative_methods_Seidel_seq {

bool IsDiagPred(float row_coeffs[], const size_t num_colls, const size_t& start_index, const size_t& index);
void copyA_(float val[], const size_t num_rows, const size_t num_colls);
float*& getA_();
void freeA_();
void setA_(float val[], const size_t num_rows, const size_t num_colls);
bool gen_matrix_with_diag_pred(const size_t num_rows, const size_t num_colls);
float gen_float_value();

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  ~TestTaskSequential();

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
  bool CheckDiagPred(float matrix[], const size_t num_rows, const size_t num_colls) const;
  bool IsQuadro(const size_t num_rows, const size_t num_colls) const;
  float* gen_vector(size_t sz);
  void iteration_perfomance();
  float d();
  void method_Seidel();
};

}  // namespace kholin_k_iterative_methods_Seidel_seq
//
#endif