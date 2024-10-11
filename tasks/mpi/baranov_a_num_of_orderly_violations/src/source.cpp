#pragma once
#include "mpi/baranov_a_num_of_orderly_violations/include/header.hpp"
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>






namespace baranov_a_num_of_orderly_violations_mpi {
    template <typename iotype, typename cntype>
    cntype baranov_a_num_of_orderly_violations_mpi::num_of_orderly_violations<iotype, cntype>::seq_proc(std::vector<iotype> vec ) {
        cntype num = 0;
        int n = vec.size();
        if (n == 0) {
          return 0;
        }
      
        if (n == 1)
        {
          return 0;
        }
        for (int i = 0; i != n-1; ++i) {

            if (vec[i + 1] < vec[i])
            {
                num++;
            }
        }
        
        return num;
    }

    //done
	template <class iotype, class cntype>
	bool baranov_a_num_of_orderly_violations_mpi::num_of_orderly_violations<iotype,cntype>::pre_processing() {
	    internal_order_test();

        int myid = world.rank();
        int world_size = world.size();
        int n;
        if (myid == 0)
        {
            n = taskData->inputs_count[0];
            input_ = std::vector<iotype>(n);
            void* ptr_r = taskData->inputs[0];
            
            void* ptr_d = input_.data();
            memcpy(ptr_d, ptr_r, sizeof(iotype) * n); //there input_ is a vector of pure data not uint8 so we can scatter to loc_vectors
            // Init value for output
           
            //to remove

            /*std::cout << std ::endl<< std ::endl;
            std::cout << "input_ is" << std::endl;
            for (int i = 0; i != n; i++) {
              std::cout << input_[i] << " ";
            }
            std::cout << std::endl<< std::endl<< std::endl;*/
            num_ = 0;

            
        }

        //for each proc we calculate size and then scatter
        broadcast(world, n, 0);
        int vec_send_size = n / world_size;
        int overflow_size = n % world_size;
        std::vector<int> send_counts(world_size, vec_send_size);
        std::vector<int> displs(world_size, 0);
        int loc_vec_size = 0;

        if (myid == 0) {
          for (int i = 0; i != world_size - 1; ++i) {
            if (i < overflow_size) {
              ++send_counts[i];
            }
            displs[i + 1] = ((send_counts[i]-1) + displs[i]);
            ++send_counts[i + 1];
          }
          loc_vec_size = send_counts[0];

        } else {
          if (myid < overflow_size) {
            ++send_counts[myid];
          }
          ++send_counts[myid];
          loc_vec_size = send_counts[myid];
        }
        
        
        loc_vec_.reserve(loc_vec_size);
        
        if (world.rank() == 0) {
          boost::mpi::scatterv(world, input_, send_counts, displs, loc_vec_.data(), loc_vec_size, 0);
        } 
        else {
          boost::mpi::scatterv(world, loc_vec_.data(), loc_vec_size, 0);
        }
        my_loc_vec_size = loc_vec_size;

        return true;
	}

	
    //done
	template <class iotype, class cntype>
    bool baranov_a_num_of_orderly_violations_mpi::num_of_orderly_violations<iotype, cntype>::run() {
        internal_order_test();
        // to remove
        /*std::cout << std ::endl << std ::endl;
        std::cout << "my_loc is  " << world.rank() << std::endl;
        auto lhs = loc_vec_.begin();
        auto rhs = loc_vec_.end();
        for (int i = 0; i != my_loc_vec_size; i++)
        {
          std::cout << loc_vec_[i] << "  ";
        }
        
        std::cout << std::endl << std::endl << std::endl;
        std::cout << "end of vectors now only chisla" << std::endl << std::endl << std::endl;*/

          
        int loc_num = 0;
        

        
        
        /*std::cout << "size is " << my_loc_vec_size << std::endl;*/
        for (int i = 0; i != my_loc_vec_size-1; ++i) {
          if (loc_vec_[i + 1] < loc_vec_[i]) {
            loc_num++;
          }
        }

        
        /*std::cout << "loc num is " << loc_num << std::endl << std::endl << std::endl;*/


        reduce(world, loc_num, num_, std::plus(), 0);
        //std::cout << std ::endl << std ::endl;
        //std::cout << "my_loc is  " << world.rank() << std::endl;
        //std::cout << num_v_lc << "  ";
        //std::cout << std::endl << std::endl << std::endl;
        //std::cout << std::endl << std::endl << std::endl;

        // to remove
        /*std::cout << "this num_ is " << num_ << std::endl << std::endl;*/
		return true;
    }


    // done
	template <class iotype, class cntype>
    bool baranov_a_num_of_orderly_violations_mpi::num_of_orderly_violations<iotype, cntype>::post_processing() {
      internal_order_test();
      
      if (world.rank() == 0)
      {
        reinterpret_cast<cntype*>(taskData->outputs[0])[0] = num_;
      }
      return true;
	} 



    //done
    template <class iotype, class cntype>
        bool baranov_a_num_of_orderly_violations_mpi::num_of_orderly_violations<iotype, cntype>::validation() {
          internal_order_test();
          // Check count elements of output
          if (world.rank() == 0) {
            if (taskData->outputs_count[0] == 1 && taskData->inputs_count.size() == 1 &&
                taskData->inputs_count[0] >= 0) {
              return true;
            }
            return false;
          }
          return true;
        }
	
}