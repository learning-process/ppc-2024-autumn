#include "mpi/anufriev_d_linear_image/include/ops_mpi_anufriev.hpp"
#include <gtest/gtest.h>
#include <boost/mpi.hpp>
#include <numeric>
#include <algorithm>

namespace anufriev_d_linear_image {

SimpleIntMPI::SimpleIntMPI(const std::shared_ptr<ppc::core::TaskData>& taskData) : Task(taskData) {}

bool SimpleIntMPI::pre_processing() {
  internal_order_test();
  return true;
}

bool SimpleIntMPI::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Проверяем корректность размеров входов/выходов
    if (!taskData || taskData->inputs.empty() || taskData->outputs.empty() || 
        taskData->inputs_count.empty() || taskData->outputs_count.empty()) {
      return false;
    }
    // Проверим, что количество входных пикселей совпадает с количеством выходных
    if (taskData->inputs_count[0] != taskData->outputs_count[0]) return false;
  }
  return true;
}

bool SimpleIntMPI::run() {
  internal_order_test();

  // Предполагается, что размер изображения задаётся снаружи, например:
  // taskData->width, taskData->height – это поля или дополнительные параметры.
  // Здесь приведён вариант, где предполагается, что в inputs_count[1] и inputs_count[2]
  // могут быть width и height. Или они могут быть где-то ещё.
  // Ниже – примерный подход. В реальном коде адаптируйте под вашу инфраструктуру.
  int root_width = 0, root_height = 0;
  if (world.rank() == 0) {
    // Например, так:
    // width_ = taskData->width;
    // height_ = taskData->height;
    // Здесь же можно сделать так, если нет других способов:
    // Пусть в inputs_count[1] хранится width, в inputs_count[2] – height.
    if (taskData->inputs_count.size() > 2) {
      root_width = static_cast<int>(taskData->inputs_count[1]);
      root_height = static_cast<int>(taskData->inputs_count[2]);
    } else {
      // Если данные не заданы, используем фиктивные
      // В реальном решении надо знать размер
      root_width = 100;  
      root_height = static_cast<int>(taskData->inputs_count[0] / root_width);
    }
  }

  // Распространим ширину и высоту на все процессы
  boost::mpi::broadcast(world, root_width, 0);
  boost::mpi::broadcast(world, root_height, 0);

  width_ = root_width;
  height_ = root_height;
  total_size_ = static_cast<size_t>(width_ * height_);

  // Загрузим входное изображение (только на 0 процессе)
  if (world.rank() == 0) {
    input_data_.resize(total_size_);
    std::copy(reinterpret_cast<int*>(taskData->inputs[0]),
              reinterpret_cast<int*>(taskData->inputs[0]) + total_size_,
              input_data_.begin());
  }

  distributeData();

  // Применяем фильтр Гаусса 3x3 к локальному куску данных
  applyGaussianFilter();

  gatherData();

  return true;
}

bool SimpleIntMPI::post_processing() {
  internal_order_test();
  // На корневом процессе скопируем результат в выходные данные
  if (world.rank() == 0) {
    std::copy(processed_data_.begin(), processed_data_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  }
  return true;
}

void SimpleIntMPI::distributeData() {
  // Вертикальное разбиение по столбцам
  int nprocs = world.size();
  int base_cols = width_ / nprocs;
  int remainder = width_ % nprocs;

  // Определяем сколько столбцов достаётся этому процессу
  int cols_for_me = base_cols + (world.rank() < remainder ? 1 : 0);

  // Определяем стартовую колонку
  int start = 0;
  for (int i = 0; i < world.rank(); i++) {
    start += base_cols + (i < remainder ? 1 : 0);
  }

  start_col_ = start;
  local_width_ = cols_for_me;

  // Рассылаем локальные данные
  // Каждый процесс должен получить local_width_ столбцов по высоте height_
  // Если cols_for_me = 0, то процесс данных не получает.
  if (world.rank() == 0) {
    // Рассылаем по другим процессам
    for (int i = 1; i < nprocs; ++i) {
      int send_cols = base_cols + (i < remainder ? 1 : 0);
      int send_start = 0;
      for (int j = 0; j < i; j++) {
        send_start += base_cols + (j < remainder ? 1 : 0);
      }
      if (send_cols > 0) {
        std::vector<int> send_buf(height_ * send_cols);
        for (int row = 0; row < height_; row++) {
          std::memcpy(&send_buf[row*send_cols],
                      &input_data_[row*width_ + send_start],
                      send_cols * sizeof(int));
        }
        world.send(i, 0, send_buf.data(), send_buf.size());
        data_path_.push_back(i);
      }
    }
    // Сами остаемся со своими столбцами в input_data_?
    // Скопируем свой кусок в input_data_ (либо используем тот же массив)
    if (local_width_ > 0) {
      std::vector<int> my_data(height_ * local_width_);
      for (int row = 0; row < height_; row++) {
        std::memcpy(&my_data[row*local_width_],
                    &input_data_[row*width_ + start_col_],
                    local_width_*sizeof(int));
      }
      input_data_.swap(my_data);
    } else {
      input_data_.clear();
    }
  } else {
    // Процессы получают свои данные
    if (local_width_ > 0) {
      input_data_.resize(height_ * local_width_);
      world.recv(0, 0, input_data_.data(), input_data_.size());
      data_path_.push_back(0);
    } else {
      input_data_.clear();
    }
  }
}

void SimpleIntMPI::gatherData() {
  int nprocs = world.size();
  int base_cols = width_ / nprocs;
  int remainder = width_ % nprocs;
  int my_cols = local_width_;

  if (world.rank() == 0) {
    processed_data_.resize(width_ * height_);
    // Копируем свои данные в итоговый массив
    if (my_cols > 0) {
      for (int row = 0; row < height_; row++) {
        std::memcpy(&processed_data_[row*width_ + start_col_],
                    &input_data_[row*my_cols],
                    my_cols*sizeof(int));
      }
    }

    // Принимаем данные от других
    for (int i = 1; i < nprocs; i++) {
      int recv_cols = base_cols + (i < remainder ? 1 : 0);
      if (recv_cols > 0) {
        int recv_start = 0;
        for (int j = 0; j < i; j++) {
          recv_start += base_cols + (j < remainder ? 1 : 0);
        }
        std::vector<int> recv_buf(height_ * recv_cols);
        world.recv(i, 0, recv_buf.data(), recv_buf.size());
        data_path_.push_back(i);
        // Копируем в итоговый массив
        for (int row = 0; row < height_; row++) {
          std::memcpy(&processed_data_[row*width_ + recv_start],
                      &recv_buf[row*recv_cols],
                      recv_cols*sizeof(int));
        }
      }
    }
  } else {
    // Отправляем данные на 0-й процесс
    if (my_cols > 0) {
      world.send(0, 0, input_data_.data(), input_data_.size());
      data_path_.push_back(0);
    }
  }
}

void SimpleIntMPI::exchangeHalo(std::vector<int>& local_data, int local_width) {
  // Для применения 3x3 фильтра нам нужен доступ к соседним столбцам.
  // Если мы имеем local_width столбцов у текущего процесса,
  // нам надо получить левый граничный столбец от левого соседа (если он есть)
  // и правый граничный столбец от правого соседа (если он есть).
  // Затем можно расширить local_data на 2 столбца (по одному с каждой стороны),
  // чтобы удобно применять фильтр.

  int left_rank = (world.rank() > 0) ? world.rank() - 1 : MPI_PROC_NULL;
  int right_rank = (world.rank() < world.size()-1) ? world.rank() + 1 : MPI_PROC_NULL;

  std::vector<int> left_col(height_), right_col(height_);
  std::vector<int> send_left_col(height_), send_right_col(height_);

  // Подготовим столбцы для отправки
  if (local_width > 0) {
    // Левый крайний столбец для отправки левому соседу
    for (int r = 0; r < height_; r++) {
      send_left_col[r] = local_data[r*local_width]; 
    }
    // Правый крайний столбец для отправки правому соседу
    for (int r = 0; r < height_; r++) {
      send_right_col[r] = local_data[r*local_width + (local_width-1)];
    }
  }

  boost::mpi::request reqs[4];
  int count = 0;
  if (left_rank != MPI_PROC_NULL) {
    reqs[count++] = world.isend(left_rank, 1, send_left_col.data(), height_);
    reqs[count++] = world.irecv(left_rank, 2, left_col.data(), height_);
  }
  if (right_rank != MPI_PROC_NULL) {
    reqs[count++] = world.isend(right_rank, 2, send_right_col.data(), height_);
    reqs[count++] = world.irecv(right_rank, 1, right_col.data(), height_);
  }

  boost::mpi::wait_all(reqs, reqs+count);

  // Добавим полученные граничные столбцы
  int extended_width = local_width;
  if (left_rank != MPI_PROC_NULL) extended_width += 1;
  if (right_rank != MPI_PROC_NULL) extended_width += 1;

  std::vector<int> extended_data(height_ * extended_width);

  int offset = (left_rank != MPI_PROC_NULL) ? 1 : 0;

  // Копируем центральную часть
  for (int r = 0; r < height_; r++) {
    std::memcpy(&extended_data[r*extended_width + offset], &local_data[r*local_width], local_width*sizeof(int));
  }

  // Копируем левый столбец, если есть
  if (left_rank != MPI_PROC_NULL) {
    for (int r = 0; r < height_; r++) {
      extended_data[r*extended_width + 0] = left_col[r];
    }
  } else {
    // Если слева нет соседей – повторим крайний столбец
    for (int r = 0; r < height_; r++) {
      extended_data[r*extended_width + 0] = local_data[r*local_width + 0];
    }
    offset = 1; // уже учтён выше
  }

  // Копируем правый столбец, если есть
  if (right_rank != MPI_PROC_NULL) {
    for (int r = 0; r < height_; r++) {
      extended_data[r*extended_width + (offset + local_width)] = right_col[r];
    }
  } else {
    // Если справа нет соседей – повторим крайний столбец
    for (int r = 0; r < height_; r++) {
      extended_data[r*extended_width + (offset + local_width)] = local_data[r*local_width + (local_width-1)];
    }
  }

  local_data.swap(extended_data);

  // Теперь в local_data есть на 1 столбец больше слева и справа (граничные)
}

void SimpleIntMPI::applyGaussianFilter() {
  if (local_width_ <= 0) return; // Нечего обрабатывать

  // Расширим данные для граничных условий по горизонтали
  std::vector<int> extended_data = input_data_;
  exchangeHalo(extended_data, local_width_);

  int extended_width = local_width_ + 2; // т.к. мы добавили слева и справа

  // Применяем свёртку
  std::vector<int> result(height_ * local_width_);

  for (int r = 0; r < height_; r++) {
    for (int c = 0; c < local_width_; c++) {
      // c+1 из-за того, что мы добавили слева 1 столбец
      int sum = 0;
      int norm = 16; // для гаусса 3x3
      for (int kr = -1; kr <= 1; kr++) {
        for (int kc = -1; kc <= 1; kc++) {
          int rr = std::min(std::max(r+kr,0), height_-1);
          int cc = c + 1 + kc; 
          sum += extended_data[rr*extended_width + cc] * kernel_[kr+1][kc+1];
        }
      }
      result[r*local_width_ + c] = sum / norm;
    }
  }

  input_data_.swap(result);
}


const std::vector<int>& SimpleIntMPI::getDataPath() const {
  return data_path_;
}

}  // namespace anufriev_d_linear_image