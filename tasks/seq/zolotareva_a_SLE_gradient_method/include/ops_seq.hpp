#pragma once
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace zolotareva_a_SLE_gradient_method_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  static void conjugate_gradient(const std::vector<double>& A, const std::vector<double>& b, std::vector<double>& x,
                                 int N);
  inline static void dot_product(double& sum, const std::vector<double>& vec1, const std::vector<double>& vec2, int n);
  inline static void matrix_vector_mult(const std::vector<double>& matrix, const std::vector<double>& vector,
                                        std::vector<double>& result, int n);
  static bool is_positive_and_simm(const double* A, int n);

 private:
  std::vector<double> A_;  // Матрица системы
  std::vector<double> b_;  // Правая часть системы
  std::vector<double> x_;  // Результирующий вектор
  int n_{0};               // Размер матрицы
};

}  // namespace zolotareva_a_SLE_gradient_method_seq