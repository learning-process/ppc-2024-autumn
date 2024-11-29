// Copyright 2023 Nesterov Alexander
#include "mpi/vladimirova_j_gather/include/ops_mpi_not_my_gather.hpp"

#include <algorithm>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "mpi/vladimirova_j_gather/include/ops_mpi.hpp"

using namespace std::chrono_literals;
using namespace vladimirova_j_gather_mpi;

bool vladimirova_j_not_my_gather_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    // Init vectors
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
  }
  // Init value for output
  return true;
}

bool vladimirova_j_not_my_gather_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool vladimirova_j_not_my_gather_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int rank = world.rank();
  int size = world.size();
  // Локальный вектор данных
  // Каждый процесс добавляет свой ранг
  // Вектор для сбора результатов (инициализируется только на процессе root)
  std::vector<int> root_vec;
  // Процесс с рангом 0 инициализирует вектор для сбора данных
  if (rank == 0) {
    root_vec.resize(size * 2 - 1);  // Инициализируем вектор для получения данных
  } else {
    local_input_.push_back(rank);
    local_input_.push_back(rank - 1);
    std::cout << std::endl << "ppp THERE " << rank << local_input_[0] << local_input_[1] << std::endl;
  }

  // Собираем данные в процессе с рангом 0

  boost::mpi::gather(world, root_vec, &local_input_, 0);

  std::cout << std::endl << "ppp end THERE " << rank << std::endl;

  // Процесс с рангом 0 выводит собранные данные
  if (rank == 0) {
    std::cout << "Gathered data: ";
    for (size_t i = 0; i < root_vec.size(); ++i) {
      std::cout << root_vec[i] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
  /*
  int r = world.rank();
  int size;
  std::vector<int> root_vec;
  if (r == 0) {
    size = input_.size() / world.size();
    root_vec.resize(5);
    for (int i = 1; i < world.size(); i++) {
      world.send(i, 0, size);
    }

    std::vector<int> pr(world.size());
    for (int i = 0; i < pr.size(); i++) {
      pr[i] = i;
    }

    std::cout << "TREE" << std::endl;
    std::for_each(pr.begin(), pr.end(), [](int number) { std::cout << number << " "; });
    std::cout << std::endl;

    pr = vladimirova_j_gather_mpi::convertToBinaryTreeOrder(pr);

    std::cout << "TREE" << std::endl;
    std::for_each(pr.begin(), pr.end(), [](int number) { std::cout << number << " "; });
    std::cout << std::endl;

    for (int i = 1; i < world.size(); i++) {
      world.send(pr[i], 0, input_.data() + size * i, size);
    }

    local_input_ = std::vector<int>(input_.begin(), input_.begin() + size);

  } else {
    world.recv(0, 0, size);
    local_input_ = std::vector<int>(size);
    world.recv(0, 0, local_input_.data(), size);
  }

  std::cout << "TREE  " << r << "   \n";
  std::for_each(local_input_.begin(), local_input_.end(), [](int number) { std::cout << number << " "; });
  std::cout << std::endl;

  //local_input_ = vladimirova_j_gather_mpi::noDeadEnds(local_input_);

  std::cout << "\n       THERE     " << r << " local input size = " << local_input_.size() << "\n";
  gather(world, local_input_, root_vec, 0);
  std::cout<<"end" << std::endl;

  std::cout << "ANS  2 " << r << "   \n";
  std::for_each(local_input_.begin(), local_input_.end(), [](int number) { std::cout << number << " "; });
  std::cout << std::endl;

  if (r == 0) {
      std::cout << "ANS  2 " << r << "   \n";
      std::for_each(root_vec.begin(), root_vec.end(), [](int number) { std::cout << number << " "; });
      std::cout << std::endl;

      root_vec.insert(root_vec.end(), input_.end() - input_.size()%world.size(), input_.end());
      std::cout << "ANS  1" << r << "   \n";
      std::for_each(root_vec.begin(), root_vec.end(), [](int number) { std::cout << number << " "; });
      std::cout << std::endl;
      root_vec = vladimirova_j_gather_mpi::noDeadEnds(root_vec);
      root_vec = vladimirova_j_gather_mpi::noStrangeSteps(root_vec);
      */
  /*
std::cout << "ANS  2 " << r << "   \n";
std::for_each(root_vec.begin(), root_vec.end(), [](int number) { std::cout << number << " "; });
std::cout << std::endl;
}
*/

  return true;
}

bool vladimirova_j_not_my_gather_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    taskData->outputs_count[0] = local_input_.size();
    auto* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(local_input_.begin(), local_input_.end(), output_data);

    // reinterpret_cast<int*>(taskData->outputs[0]) = res;
  }
  return true;
}
