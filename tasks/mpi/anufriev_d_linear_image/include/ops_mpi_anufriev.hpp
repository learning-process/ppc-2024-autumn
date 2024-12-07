#pragma once

#include <boost/mpi.hpp>
#include <vector>
#include <algorithm>
#include <cstring>
#include <mpi.h>
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
  void exchangeHalo(std::vector<int>& local_data, int local_width);

  boost::mpi::communicator world;

  std::vector<int> original_data_;      // Данные на 0-м процессе в column-major порядке
  std::vector<int> local_data_;         // Локальные данные на каждом процессе
  std::vector<int> processed_data_;     // Сконкатенированные результаты на 0-м процессе

  size_t total_size_ = 0;
  int width_ = 0;
  int height_ = 0;

  // Индексы для распределения
  int start_col_ = 0;
  int local_width_ = 0;

  std::vector<int> data_path_;

  const int kernel_[3][3] = {
    {1, 2, 1},
    {2, 4, 2},
    {1, 2, 1}
  };
};

}  // namespace anufriev_d_linear_image