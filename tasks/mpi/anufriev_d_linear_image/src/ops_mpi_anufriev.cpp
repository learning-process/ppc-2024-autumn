#include "mpi/anufriev_d_linear_image/include/ops_mpi_anufriev.hpp"
#include <gtest/gtest.h>
#include <boost/mpi.hpp>
#include <numeric>
#include <algorithm>
#include <mpi.h>
#include <iostream> // Для отладочных сообщений (при необходимости)

namespace anufriev_d_linear_image {

SimpleIntMPI::SimpleIntMPI(const std::shared_ptr<ppc::core::TaskData>& taskData) : Task(taskData) {}

bool SimpleIntMPI::pre_processing() {
    internal_order_test(); // Предполагается, что этот метод определён в базовом классе
    return true;
}

bool SimpleIntMPI::validation() {
    internal_order_test(); // Предполагается, что этот метод определён в базовом классе
    if (world.rank() == 0) {
        if (!taskData || taskData->inputs.size() < 3 || taskData->inputs_count.size() < 3 ||
            taskData->outputs.empty() || taskData->outputs_count.empty()) {
            std::cerr << "Validation failed: Недостаточно входных или выходных данных.\n";
            return false;
        }

        width_ = *reinterpret_cast<int*>(taskData->inputs[1]);
        height_ = *reinterpret_cast<int*>(taskData->inputs[2]);

        size_t expected_size = static_cast<size_t>(width_ * height_ * sizeof(int));

        if (width_ < 3 || height_ < 3) {
            std::cerr << "Validation failed: width или height меньше 3.\n";
            return false;
        }

        if (taskData->inputs_count[0] != expected_size) {
            std::cerr << "Validation failed: inputs_count[0] != width * height * sizeof(int).\n";
            std::cerr << "Expected: " << expected_size << ", Got: " << taskData->inputs_count[0] << "\n";
            return false;
        }

        if (taskData->outputs_count[0] != expected_size) {
            std::cerr << "Validation failed: outputs_count[0] != width * height * sizeof(int).\n";
            std::cerr << "Expected: " << expected_size << ", Got: " << taskData->outputs_count[0] << "\n";
            return false;
        }

        // Перестраиваем данные в row-major порядок
        original_data_.resize(width_ * height_);
        int* input_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
        std::copy(input_ptr, input_ptr + (width_ * height_), original_data_.begin());
    }

    // Распространяем width и height на все процессы
    boost::mpi::broadcast(world, width_, 0);
    boost::mpi::broadcast(world, height_, 0);
    total_size_ = static_cast<size_t>(width_ * height_);

    return true;
}

bool SimpleIntMPI::run() {
    internal_order_test(); // Предполагается, что этот метод определён в базовом классе

    distributeData();
    exchangeHalo();
    applyGaussianFilter();
    gatherData();

    return true;
}

bool SimpleIntMPI::post_processing() {
    internal_order_test(); // Предполагается, что этот метод определён в базовом классе
    if (world.rank() == 0) {
        // Данные уже собраны в row-major порядке
        int* output_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
        std::copy(processed_data_.begin(), processed_data_.end(), output_ptr);
    }
    return true;
}

void SimpleIntMPI::distributeData() {
    MPI_Comm comm = world;
    int nprocs = world.size();
    int rank = world.rank();

    // Определяем количество строк на каждом процессе
    int base_rows = height_ / nprocs;
    int remainder = height_ % nprocs;

    std::vector<int> sendcounts(nprocs);
    std::vector<int> displs(nprocs);

    for (int i = 0; i < nprocs; ++i) {
        sendcounts[i] = (base_rows + (i < remainder ? 1 : 0)) * width_;
        displs[i] = (i < remainder) ? i * (base_rows + 1) * width_
                                    : remainder * (base_rows + 1) * width_ + (i - remainder) * base_rows * width_;
    }

    local_height_ = base_rows + (rank < remainder ? 1 : 0);
    start_row_ = (rank < remainder) ? rank * (base_rows + 1)
                                    : remainder * (base_rows + 1) + (rank - remainder) * base_rows;

    // Добавляем гало-области (по одной строке сверху и снизу)
    int halo_rows = 2;
    local_data_.resize((local_height_ + halo_rows) * width_, 0);

    // Получаем собственные данные
    MPI_Scatterv(world.rank() == 0 ? original_data_.data() : nullptr,
                 sendcounts.data(), displs.data(), MPI_INT,
                 &local_data_[width_], // Пропускаем первую строку для верхнего гало
                 local_height_ * width_, MPI_INT,
                 0, comm);
}

void SimpleIntMPI::exchangeHalo() {
    MPI_Comm comm = world;
    int rank = world.rank();
    int nprocs = world.size();

    // Определяем соседей
    int up = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int down = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;

    // Буферы для обмена
    std::vector<int> send_up(width_);
    std::vector<int> send_down(width_);
    std::vector<int> recv_up(width_);
    std::vector<int> recv_down(width_);

    // Подготавливаем данные для отправки
    if (local_height_ > 0) {
        std::copy(&local_data_[width_], &local_data_[2 * width_], send_up.begin()); // Первая собственная строка
        std::copy(&local_data_[(local_height_) * width_], &local_data_[(local_height_ + 1) * width_], send_down.begin()); // Последняя собственная строка
    }

    MPI_Request reqs[4];
    int req_count = 0;

    // Отправка вверх и получение сверху
    if (up != MPI_PROC_NULL) {
        MPI_Isend(send_up.data(), width_, MPI_INT, up, 0, comm, &reqs[req_count++]);
        MPI_Irecv(recv_up.data(), width_, MPI_INT, up, 1, comm, &reqs[req_count++]);
    } else {
        // Дублирование верхней строки для верхнего гало
        std::copy(send_up.begin(), send_up.end(), recv_up.begin());
    }

    // Отправка вниз и получение снизу
    if (down != MPI_PROC_NULL) {
        MPI_Isend(send_down.data(), width_, MPI_INT, down, 1, comm, &reqs[req_count++]);
        MPI_Irecv(recv_down.data(), width_, MPI_INT, down, 0, comm, &reqs[req_count++]);
    } else {
        // Дублирование нижней строки для нижнего гало
        std::copy(send_down.begin(), send_down.end(), recv_down.begin());
    }

    // Ожидание завершения всех операций
    if (req_count > 0) {
        MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
    }

    // Заполнение гало-областей
    if (up != MPI_PROC_NULL) {
        std::copy(recv_up.begin(), recv_up.end(), local_data_.begin()); // Верхний гало
    } else {
        std::copy(send_up.begin(), send_up.end(), local_data_.begin()); // Дублируем первую строку
    }

    if (down != MPI_PROC_NULL) {
        std::copy(recv_down.begin(), recv_down.end(), &local_data_[(local_height_ + 1) * width_]); // Нижний гало
    } else {
        std::copy(send_down.begin(), send_down.end(), &local_data_[(local_height_ + 1) * width_]); // Дублируем последнюю строку
    }
}

void SimpleIntMPI::applyGaussianFilter() {
    std::vector<int> result(local_height_ * width_, 0);

    for (int r = 1; r <= local_height_; r++) { // r=1..local_height_
        for (int c = 0; c < width_; c++) {
            int sum = 0;
            for (int kr = -1; kr <= 1; kr++) {
                for (int kc = -1; kc <= 1; kc++) {
                    int rr = r + kr;
                    int cc = std::min(std::max(c + kc, 0), width_ - 1); // Клэмпинг по горизонтали

                    sum += local_data_[rr * width_ + cc] * kernel_[kr + 1][kc + 1];
                }
            }
            result[(r - 1) * width_ + c] = sum / 16;
        }
    }

    // Перезаписываем собственные строки результатами
    std::copy(result.begin(), result.end(), &local_data_[width_]);
}

void SimpleIntMPI::gatherData() {
    MPI_Comm comm = world;
    int nprocs = world.size();
    //int rank = static_cast<int>(world.rank());

    // Определяем количество строк на каждом процессе
    int base_rows = height_ / nprocs;
    int remainder = height_ % nprocs;

    std::vector<int> recvcounts(nprocs);
    std::vector<int> displs(nprocs);

    for (int i = 0; i < nprocs; ++i) {
        recvcounts[i] = (base_rows + (i < remainder ? 1 : 0)) * width_;
        displs[i] = (i < remainder) ? i * (base_rows + 1) * width_
                                    : remainder * (base_rows + 1) * width_ + (i - remainder) * base_rows * width_;
    }

    if (world.rank() == 0) {
        processed_data_.resize(width_ * height_);
    }

    MPI_Gatherv(&local_data_[width_], local_height_ * width_, MPI_INT,
                world.rank() == 0 ? processed_data_.data() : nullptr,
                recvcounts.data(), displs.data(), MPI_INT,
                0, comm);
}

const std::vector<int>& SimpleIntMPI::getDataPath() const {
    return data_path_;
}

}  // namespace anufriev_d_linear_image