#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace list_ops {
enum ops_ { METHOD_SEIDEL };
}

namespace kholin_k_iterative_methods_Seidel_mpi {

bool IsDiagPred(float row_coeffs[], const size_t num_colls, const size_t& start_index, const size_t& index);
void copyA_(float val[], const size_t num_rows, const size_t num_colls);
float*& getA_();
void setA_(float val[], const size_t num_rows, const size_t num_colls);
bool gen_matrix_with_diag_pred(const size_t num_rows, const size_t num_colls);
float gen_float_value();

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_, list_ops::ops_ op_)
      : Task(std::move(taskData_)), op(op_) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  ~TestMPITaskSequential();

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
  list_ops::ops_ op;
};

class TestMPITaskParallel : public ppc::core::Task {
  MPI_Datatype get_mpi_type();

 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_, list_ops::ops_ op_)
      : Task(std::move(taskData_)), op(op_) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  ~TestMPITaskParallel();

 private:
  float* A;
  float* X;
  float* X_next;
  float* X_prev;
  float* X0;
  float* B;
  float* C;
  float* upper_C;
  float* lower_C;
  float epsilon;
  int* upper_send_counts;
  int* lower_send_counts;
  int* local_upper_counts;
  int* local_lower_counts;
  int upper_proc_size;
  int* upper_displs;
  int* lower_displs;
  size_t n_rows;
  size_t n_colls;
  int count;
  float max_delta;
  float global_x;
  void SetDefault();
  bool CheckDiagPred(float matrix[], const size_t num_rows, const size_t num_colls) const;
  bool IsQuadro(const size_t num_rows, const size_t num_colls) const;
  float* gen_vector(size_t sz);
  void to_upper_diag_matrix();
  void to_lower_diag_matrix();
  void iteration_perfomance();
  float d();
  void method_Seidel();
  list_ops::ops_ op;
  MPI_Datatype sz_t;
};

}  // namespace kholin_k_iterative_methods_Seidel_mpi