#pragma once

#include <boost/mpi.hpp>
#include <vector>
#include <algorithm>
#include <cstring>
#include "core/task/include/task.hpp"

namespace anufriev_d_linear_image {

class SimpleIntMPI : public ppc::core::Task {
 public:
  explicit SimpleIntMPI(const std::shared_ptr<ppc::core::TaskData>& taskData);

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  const std::vector<int>& getDataPath() const;

 private:
  void distributeData();
  void gatherData();
  void applyGaussianFilter();

  // Обмен граничными столбцами между соседними процессами
  void exchangeHalo(std::vector<int>& local_data, int local_width);

  boost::mpi::communicator world;
  std::vector<int> input_data_;
  std::vector<int> processed_data_;

  size_t total_size_;
  int width_;
  int height_;

  std::vector<int> data_path_;

  // Индексы для распределения по столбцам
  int start_col_;
  int local_width_;

  // Гауссово ядро 3x3 с нормализацией 1/16:
  // [1 2 1
  //  2 4 2
  //  1 2 1]
  const int kernel_[3][3] = {
    {1, 2, 1},
    {2, 4, 2},
    {1, 2, 1}
  };
};

}  // namespace anufriev_d_linear_image