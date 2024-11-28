// ops_mpi.hpp
#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace kharin_m_seidel_method {

class GaussSeidelSequential : public ppc::core::Task {
 public:
  explicit GaussSeidelSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double** a = nullptr;  // Матрица коэффициентов
  double* b = nullptr;   // Вектор свободных членов
  double* x = nullptr;   // Вектор решений
  double* p = nullptr;   // Предыдущее приближение
  int n = 0;             // Размерность системы
  double eps = 0.0;      // Точность вычислений
};

}  // namespace kharin_m_seidel_method