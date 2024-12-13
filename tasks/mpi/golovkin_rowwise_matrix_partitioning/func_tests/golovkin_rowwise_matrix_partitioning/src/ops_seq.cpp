// Golovkin Maksim

#include "seq/golovkin_rowwise_matrix_partitioning/include/ops_seq.hpp"

#include <vector>
#include <cstdint>
#include <memory>
#include "core/task/include/task.hpp"
using namespace golovkin_rowwise_matrix_partitioning;

MatrixMultiplicationTask::MatrixMultiplicationTask(const std::shared_ptr<ppc::core::TaskData>& taskData)
    : ppc::core::Task(taskData), taskData_(taskData) {}

  // Метод валидации данных
bool MatrixMultiplicationTask:: validation() {
     internal_order_test();
    if (!taskData_ || taskData_->inputs.size() < 2 || taskData_->outputs.size() < 1) {
      return false;
    }

    auto* matrixA = reinterpret_cast<std::vector<std::vector<double>>*>(taskData_->inputs[0]);
    auto* matrixB = reinterpret_cast<std::vector<std::vector<double>>*>(taskData_->inputs[1]);
    if (!matrixA || !matrixB || matrixA->empty() || matrixB->empty()) {
      return false;
    }

    // Проверяем совместимость размеров матриц
    if (matrixA->at(0).size() != matrixB->size()) {
      return false;
    }

    return true;
  }

  // Метод предобработки данных
bool MatrixMultiplicationTask::pre_processing() {
    internal_order_test();
    // Проверяем размеры матриц и выделяем память для результата
    auto* matrixA = reinterpret_cast<std::vector<std::vector<double>>*>(taskData_->inputs[0]);
    auto* matrixB = reinterpret_cast<std::vector<std::vector<double>>*>(taskData_->inputs[1]);
    auto* result = reinterpret_cast<std::vector<std::vector<double>>*>(taskData_->outputs[0]);

    if (!matrixA || !matrixB || !result) {
      return false;
    }

    size_t rowsA = matrixA->size();
    size_t colsB = matrixB->at(0).size();
    result->resize(rowsA, std::vector<double>(colsB, 0.0));
    return true;
  }

  // Метод умножения матриц
bool MatrixMultiplicationTask::multiplier(std::vector<std::vector<double>>& matrixA,
    std::vector<std::vector<double>>& matrixB,
    std::vector<std::vector<double>>& result) {
    
    size_t rowsA = matrixA.size();
    size_t colsA = matrixA[0].size();
    size_t colsB = matrixB[0].size();

    for (size_t i = 0; i < rowsA; ++i) {
      for (size_t j = 0; j < colsB; ++j) {
        for (size_t k = 0; k < colsA; ++k) {
          result[i][j] += matrixA[i][k] * matrixB[k][j];
        }
      }
    }
    return true;
  }

  // Метод выполнения
bool MatrixMultiplicationTask:: run() {
    internal_order_test();
    auto* matrixA = reinterpret_cast<std::vector<std::vector<double>>*>(taskData_->inputs[0]);
    auto* matrixB = reinterpret_cast<std::vector<std::vector<double>>*>(taskData_->inputs[1]);
    auto* result = reinterpret_cast<std::vector<std::vector<double>>*>(taskData_->outputs[0]);

    if (!matrixA || !matrixB || !result) {
      return false;
    }

    // Вызов метода умножения
    return multiplier(*matrixA, *matrixB, *result);
  }

  // Метод постобработки данных
bool MatrixMultiplicationTask::post_processing() {
    internal_order_test();
    if (!result_.empty()) {
      // Порог для устранения ошибок с плавающей точкой
      const double epsilon = 1e-9;

      // Проходим по всем элементам матрицы результата
      for (size_t i = 0; i < result_.size(); ++i) {
        for (size_t j = 0; j < result_[0].size(); ++j) {
          // Если значение близко к нулю, устанавливаем его в 0
          if (std::abs(result_[i][j]) < epsilon) {
            result_[i][j] = 0.0;
          }
        }
      }

      return true;  // Постобработка завершена успешно
    }
    return false;  // Если матрица пуста, возвращаем ошибку
  }
