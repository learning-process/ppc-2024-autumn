#pragma once

 #include <gtest/gtest.h>

 #include <boost/mpi/collectives.hpp>
 #include <boost/mpi/communicator.hpp>
 #include <memory>
 #include <numeric>
 #include <queue>
 #include <string>
 #include <utility>
 #include <vector>

 #include "core/task/include/task.hpp"

 namespace dudchenko_o_sleeping_barber_mpi {

 class TestMPISleepingBarber : public ppc::core::Task {
  public:
   explicit TestMPISleepingBarber(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

   bool pre_processing() override;
   bool validation() override;
   bool run() override;
   bool post_processing() override;

  private:
   int max_wait{};
   int result{};
   boost::mpi::communicator world;

   void next_client(int client);
 };
 }  // namespace dudchenko_o_sleeping_barber_mpi