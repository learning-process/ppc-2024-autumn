#include "mpi/pikarychev_i_sleeping_barber/include/ops_mpi.hpp"

#include <random>
#include <thread>
#include <vector>

#include "boost/mpi/communicator.hpp"

using namespace std::chrono_literals;

enum Ranks : int { RankCoordinator = 0, RankBarber = 1 };  // NOLINT(performance-enum-size)
enum Tags : int {                                          // NOLINT(performance-enum-size)
  JoinWaitingTag,
  IncomingCustomerTag,
  AcceptingCustomerTag,
  ReleasingBarberTag
};

constexpr int BarberTerminationCustomerMagicId = -1;

bool pikarychev_i_sleeping_barber_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    capacity = *reinterpret_cast<int*>(taskData->inputs[0]);
  }
  return true;
}

bool pikarychev_i_sleeping_barber_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  return world.size() > 2 && (world.rank() != 0 || (taskData->inputs.size() == 1 && taskData->inputs_count[0] == 1));
}

bool pikarychev_i_sleeping_barber_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  std::random_device dev;
  std::mt19937 gen(dev());

  boost::mpi::broadcast(world, capacity, 0);
  const auto customers = world.size() - 2;
  const auto rank = world.rank();

  switch (rank) {
    case RankCoordinator: {
      int i = 0;
      int waiting = 0;
      while (i < customers) {
        const auto status = world.recv(boost::mpi::any_source, boost::mpi::any_tag);
        const auto& src = status.source();
        const auto& tag = status.tag();
        switch (tag) {
          case JoinWaitingTag: {
            world.isend(src, tag, waiting);
            printf("Coordinator: accept %d\n", src);
            if (waiting >= capacity) {
              ++i;
              printf("Coordinator: drop %d\n", src);
              continue;
            }
            ++waiting;
            printf("Coordinator: push %d\n", src);
            world.isend(RankBarber, IncomingCustomerTag, src);
            break;
          }
          case AcceptingCustomerTag: {
            ++i;
            --waiting;
            printf("Coordinator: pop\n");
            break;
          }
        }
      }
      printf("Coordinator: terminating barber\n");
      world.send(RankBarber, IncomingCustomerTag, BarberTerminationCustomerMagicId);
      break;
    }
    case RankBarber: {
      while (true) {
        int customer;
        world.recv(RankCoordinator, IncomingCustomerTag, customer);
        printf("Barber: accept %d\n", customer);

        if (customer == BarberTerminationCustomerMagicId) {
          printf("Barber: terminating\n");
          break;
        }

        printf("Barber: notify %d\n", customer);
        world.send(RankCoordinator, AcceptingCustomerTag);

        world.send(customer, AcceptingCustomerTag);
        world.recv(customer, ReleasingBarberTag);
        std::this_thread::sleep_for(std::chrono::milliseconds(5 + gen() % 5));
        world.send(customer, ReleasingBarberTag);

        printf("Barber: done %d\n", customer);
      }
      break;
    }
    default: {
      std::this_thread::sleep_for(std::chrono::milliseconds(10 + gen() % 20));

      int waiting;
      printf("Customer#%d: acquiring coordinator\n", rank);
      world.send(RankCoordinator, JoinWaitingTag);
      world.recv(RankCoordinator, JoinWaitingTag, waiting);
      if (waiting >= capacity) {
        printf("Customer#%d: leave, queue is full\n", rank);
        break;
      }

      world.recv(RankBarber, AcceptingCustomerTag);
      std::this_thread::sleep_for(std::chrono::milliseconds(10 + gen() % 10));
      world.send(RankBarber, ReleasingBarberTag);
      world.recv(RankBarber, ReleasingBarberTag);
      printf("Customer#%d: complete\n", rank);

      break;
    }
  }

  world.barrier();

  return true;
}

bool pikarychev_i_sleeping_barber_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  return true;
}
