// Copyright 2024 Sotskov Alexander
#pragma once

#include <boost/mpi.hpp>
#include <memory>
#include <string>
#include <vector>
#include "core/task/include/task.hpp" // Включение вашего базового класса Task

namespace sotskov_a_sum_element_matrix_mpi {

// Функция для генерации случайной матрицы
std::vector<std::vector<int>> getRandomMatrix(int rows, int cols);

// Последовательный класс задачи
class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<int>> matrix_; // Входная матрица
  std::vector<std::vector<int>> input_; // Входная матрица
  int res; // Результат
};

// Параллельный класс задачи с использованием MPI
class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_, std::string ops_)
      : Task(std::move(taskData_)), ops(std::move(ops_)), world(boost::mpi::communicator()) {} // Properly initialize 'world'

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<int>> input_, local_input_; // Входная и локальная матрицы
  int res; // Результат
  std::string ops; // Операции
  boost::mpi::communicator world; // MPI коммуникатор
};

}  // namespace sotskov_a_sum_element_matrix_mpi
