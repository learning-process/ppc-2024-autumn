#pragma once

#include <gtest/gtest.h>
#include <memory.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <utility>

#include "core/task/include/task.hpp"

namespace list_ops {
enum ops_ : std::uint8_t { METHOD_SEIDEL };
}

namespace kholin_k_iterative_methods_Seidel_mpi {
std::vector<float> gen_matrix_with_diag_pred(size_t num_rows, size_t num_colls, float p1, float p2);
float gen_float_value();
bool IsDiagPred(std::vector<float> row_coeffs, size_t num_colls, size_t start_index, size_t index);
std::vector<float> gen_vector(size_t sz);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_, list_ops::ops_ op_)
      : Task(std::move(taskData_)), op(op_) {}
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
  static int rank(std::vector<float> local_matrix, size_t n, size_t m);
  void iteration_perfomance();
  void SetDefault();
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
  ~TestMPITaskParallel() override;

 private:
  std::vector<float> A;
  std::vector<float> X;
  std::vector<float> X_next;
  std::vector<float> X_prev;
  std::vector<float> X0;
  std::vector<float> B;
  std::vector<float> C;
  std::vector<float> upper_C;
  std::vector<float> lower_C;
  float epsilon;
  std::vector<float> upper_send_counts;
  std::vector<float> lower_send_counts;
  std::vector<float> local_upper_counts;
  std::vector<float> local_lower_counts;
  std::vector<float> upper_displs;
  std::vector<float> lower_displs;
  size_t n_rows;
  size_t n_colls;
  int count;
  float max_delta;
  float global_x;
  static bool CheckDiagPred(float matrix[], size_t num_rows, size_t num_colls);
  static bool IsQuadro(size_t num_rows, size_t num_colls);
  static int rank(std::vector<float> local_matrix, size_t n, size_t m);
  static std::vector<float> gen_vector(size_t sz);
  void to_upper_diag_matrix();
  void to_lower_diag_matrix();
  void SetDefault();
  void iteration_perfomance();
  float d();
  void method_Seidel();
  list_ops::ops_ op;
  MPI_Datatype sz_t;
};

}  // namespace kholin_k_iterative_methods_Seidel_mpi