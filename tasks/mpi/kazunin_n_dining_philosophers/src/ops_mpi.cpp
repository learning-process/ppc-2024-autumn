#include "mpi/kazunin_n_dining_philosophers/include/ops_mpi.hpp"

#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>

bool kazunin_n_dining_philosophers_mpi::DiningPhilosophersParallelMPI::validation() {
  // internal_order_test();

  return true;
}

bool kazunin_n_dining_philosophers_mpi::DiningPhilosophersParallelMPI::pre_processing() {
  // internal_order_test();

  return true;
}

bool kazunin_n_dining_philosophers_mpi::DiningPhilosophersParallelMPI::run() {
  // internal_order_test();

  int rank;  // Идентификатор процесса
  int size;  // Общее количество процессов

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int NUM_PHILOSOPHERS = size;

  // Инициализация генератора случайных чисел
  std::srand(std::time(nullptr) + rank);

  // Определение соседей
  int leftNeighbor = (rank + NUM_PHILOSOPHERS - 1) % NUM_PHILOSOPHERS;
  int rightNeighbor = (rank + 1) % NUM_PHILOSOPHERS;

  // Каждый философ начинает без вилок
  bool hasLeftFork = false;
  bool hasRightFork = false;

  // Вилки находятся между философами
  bool leftForkAvailable = rank > leftNeighbor;
  bool rightForkAvailable = rank > rightNeighbor;

  State state = THINKING;

  MPI_Status status;

  // Временные переменные
  int thinkTime = rand() % 3 + 1;
  int eatTime = rand() % 3 + 1;
  int timeCounter = 0;

  // Время симуляции
  const int SIMULATION_TIME = 5;
  double startTime = MPI_Wtime();

  while (MPI_Wtime() - startTime < SIMULATION_TIME) {
    // Проверка входящих сообщений
    int flag;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);

    if (flag != 0) {
      int message;
      MPI_Recv(&message, 1, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &status);

      if (status.MPI_TAG == REQUEST_FORK) {
        // Обработка запроса вилки
        if (status.MPI_SOURCE == leftNeighbor && leftForkAvailable) {
          leftForkAvailable = false;
          message = 0;
          MPI_Send(&message, 1, MPI_INT, status.MPI_SOURCE, FORK_AVAILABLE, MPI_COMM_WORLD);
          std::cout << "Философ " << rank << " передал левую вилку философу " << leftNeighbor << std::endl;
        } else if (status.MPI_SOURCE == rightNeighbor && rightForkAvailable) {
          rightForkAvailable = false;
          message = 0;
          MPI_Send(&message, 1, MPI_INT, status.MPI_SOURCE, FORK_AVAILABLE, MPI_COMM_WORLD);
          std::cout << "Философ " << rank << " передал правую вилку философу " << rightNeighbor << std::endl;
        }
      } else if (status.MPI_TAG == FORK_AVAILABLE) {
        // Получение вилки
        if (status.MPI_SOURCE == leftNeighbor) {
          hasLeftFork = true;
          std::cout << "Философ " << rank << " получил левую вилку от " << leftNeighbor << std::endl;
        } else if (status.MPI_SOURCE == rightNeighbor) {
          hasRightFork = true;
          std::cout << "Философ " << rank << " получил правую вилку от " << rightNeighbor << std::endl;
        }
      }
    }

    // Действия философа
    if (state == THINKING) {
      // Философ думает
      if (timeCounter < thinkTime) {
        if (timeCounter == 0) {
          std::cout << "Философ " << rank << " думает в течение " << thinkTime << " секунд." << std::endl;
        }
        sleep(1);
        timeCounter++;
      } else {
        // Переход к состоянию "голоден"
        state = HUNGRY;
        timeCounter = 0;
        thinkTime = rand() % 3 + 1;
      }
    } else if (state == HUNGRY) {
      // Запрос вилок
      if (!hasLeftFork) {
        int message = 0;
        MPI_Send(&message, 1, MPI_INT, leftNeighbor, REQUEST_FORK, MPI_COMM_WORLD);
        // std::cout << "Философ " << rank << " запрашивает левую вилку у " << leftNeighbor << std::endl;
      }
      if (!hasRightFork) {
        int message = 0;
        MPI_Send(&message, 1, MPI_INT, rightNeighbor, REQUEST_FORK, MPI_COMM_WORLD);
        // std::cout << "Философ " << rank << " запрашивает правую вилку у " << rightNeighbor << std::endl;
      }
      // Проверка наличия обеих вилок
      if (hasLeftFork && hasRightFork) {
        state = EATING;
        eatTime = rand() % 3 + 1;
        std::cout << "Философ " << rank << " начинает есть в течение " << eatTime << " секунд." << std::endl;
        timeCounter = 0;
      }
    } else if (state == EATING) {
      // Философ ест
      if (timeCounter < eatTime) {
        sleep(1);
        timeCounter++;
      } else {
        // Переход к состоянию "думает"
        state = THINKING;
        timeCounter = 0;
        std::cout << "Философ " << rank << " закончил есть." << std::endl;

        // Возврат вилок соседям
        int message = 0;

        // Возврат левой вилки
        hasLeftFork = false;
        leftForkAvailable = true;
        if (rank != leftNeighbor) {
          MPI_Send(&message, 1, MPI_INT, leftNeighbor, FORK_AVAILABLE, MPI_COMM_WORLD);
          std::cout << "Философ " << rank << " вернул левую вилку философу " << leftNeighbor << std::endl;
        }

        // Возврат правой вилки
        hasRightFork = false;
        rightForkAvailable = true;
        if (rank != rightNeighbor) {
          MPI_Send(&message, 1, MPI_INT, rightNeighbor, FORK_AVAILABLE, MPI_COMM_WORLD);
          std::cout << "Философ " << rank << " вернул правую вилку философу " << rightNeighbor << std::endl;
        }

        // Сброс временных переменных
        thinkTime = rand() % 3 + 1;
      }
    }
  }

  return true;
}

bool kazunin_n_dining_philosophers_mpi::DiningPhilosophersParallelMPI::post_processing() {
  // internal_order_test();

  return true;
}
