// ops_mpi.hpp
#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
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
  std::vector<double> a;       // Матрица коэффициентов в линейной форме
  std::vector<double> b;       // Вектор свободных членов
  std::vector<double> x;       // Вектор решений
  std::vector<double> p;       // Предыдущее приближение
  int n = 0;                   // Размерность системы
  double eps = 0.0;            // Точность вычислений
  int max_iterations = 10000;  // Максимальное количество итераций
};

class GaussSeidelParallel : public ppc::core::Task {
 public:
  explicit GaussSeidelParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> a;           // Матрица коэффициентов в линейной форме
  std::vector<double> b;           // Вектор свободных членов
  std::vector<double> x;           // Вектор решений
  std::vector<double> p;           // Предыдущее приближение
  int n = 0;                       // Размерность системы
  double eps = 0.0;                // Точность вычислений
  int max_iterations = 10000;      // Максимальное количество итераций
  boost::mpi::communicator world;  // MPI коммуникатор
};

}  // namespace kharin_m_seidel_method