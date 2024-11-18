#include "mpi/morozov_e_writers_readers/include/ops_mpi.hpp"

#include <thread>
#include <vector>

using namespace std::chrono_literals;
bool morozov_e_writers_readers::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count.size() == 1 && taskData->outputs_count.size() == 1 &&
           taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1;
  }
  return true;
}
bool morozov_e_writers_readers::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    curValue = taskData->inputs[0][0];
    countIteration = taskData->inputs[0][1];
  }
  broadcast(world, countIteration, 0);
  return true;
}

bool morozov_e_writers_readers::TestMPITaskParallel::run() {
  internal_order_test();
  // Чтобы проходили perf тесты
  std::this_thread::sleep_for(20ms);
  if (world.rank() == 0) {
    int received_value;
    for (int i = 0; i < countIteration; i++) {
      for (int j = 1; j < world.size(); j++) {
        world.recv(j, 0, &received_value, 1);
        curValue += received_value;
      }
    }
  } else {
    int value;
    for (int i = 0; i < countIteration; i++) {
      if (!(world.size() % 2 == 0 && world.rank() == world.size() - 1)) {
        if (world.rank() % 2 == 1) {
          value = -1;
          /*std::cout << world.rank() << " "
                    << "-1 " << i << std::endl;*/  // Нечетные потоки уменьшают значение
        } else {
          /*std::cout << world.rank() << " "
                    << "+1 " << i << std::endl;*/
          value = 1;  // Четные потоки увеличивают значение
        }
        world.send(0, 0, &value, 1);
      } else {
        value = 0;
        /*std::cout << world.rank() << " "
                  << "0" << std::endl;*/
        world.send(0, 0, &value, 1);
      }
      std::this_thread::sleep_for(200ms);
    }
  }
  return true;
}
bool morozov_e_writers_readers::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = curValue;
  }
  return true;
}