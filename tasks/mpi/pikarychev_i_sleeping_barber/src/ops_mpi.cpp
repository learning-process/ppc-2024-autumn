#include "mpi/pikarychev_i_sleeping_barber/include/ops_mpi.hpp"

#include <random>
#include <thread>
#include <vector>

#include "boost/mpi/communicator.hpp"

using namespace std::chrono_literals;

enum Ranks : uint8_t { RankCoordinator = 0, RankBarber = 1 };
enum Tags : uint8_t { JoinWaitingTag, IncomingCustomerTag, AcceptingCustomerTag, ReleasingBarberTag };

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
            if (waiting >= capacity) {
              ++i;
              continue;
            }
            ++waiting;
            world.isend(RankBarber, IncomingCustomerTag, src);
            break;
          }
          case AcceptingCustomerTag: {
            ++i;
            --waiting;
            break;
          }
        }
      }
      world.send(RankBarber, IncomingCustomerTag, BarberTerminationCustomerMagicId);
      break;
    }
    case RankBarber: {
      while (true) {
        int customer;
        world.recv(RankCoordinator, IncomingCustomerTag, customer);

        if (customer == BarberTerminationCustomerMagicId) {
          break;
        }

        world.send(RankCoordinator, AcceptingCustomerTag);

        world.send(customer, AcceptingCustomerTag);
        world.recv(customer, ReleasingBarberTag);
        std::this_thread::sleep_for(std::chrono::milliseconds(5 + gen() % 5));
        world.send(customer, ReleasingBarberTag);
      }
      break;
    }
    default: {
      std::this_thread::sleep_for(std::chrono::milliseconds(10 + gen() % 20));

      int waiting;
      world.send(RankCoordinator, JoinWaitingTag);
      world.recv(RankCoordinator, JoinWaitingTag, waiting);
      if (waiting >= capacity) {
        break;
      }

      world.recv(RankBarber, AcceptingCustomerTag);
      std::this_thread::sleep_for(std::chrono::milliseconds(10 + gen() % 10));
      world.send(RankBarber, ReleasingBarberTag);
      world.recv(RankBarber, ReleasingBarberTag);

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
