#include <iostream>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <memory>
#include <queue>
#include "CartpoleEnv.h"
#include <cstdint>
#include <tuple>
#include <fstream>

using namespace std;

class DQNAgent {
public:
    DQNAgent(double learning_rate, double epsilon_start, double epsilon_end, double epsilon_decay, double tau)
    :learning_rate_(learning_rate),epsilon_start_(epsilon_start), epsilon_end_(epsilon_end),
    epsilon_decay_(epsilon_decay),tau_(tau), device_("cpu")
     {
        
        main_net_ = nn::Sequential(
            nn::Linear(env.get_state().size(), 128),
            nn::ReLU(),
            nn::Linear(128, 128),
            nn::ReLU(),
            nn::Linear(128, env.get_action_space())
        );
        target_net_ = nn::Sequential(
            nn::Linear(env.get_state().size(), 128),
            nn::ReLU(),
            nn::Linear(128, 128),
            nn::ReLU(),
            nn::Linear(128, env.get_action_space())
        );
        optimizer = std::make_unique<torch::optim::AdamW>(main_net_->parameters(), learning_rate);
        main_net_->to(device_);
    }

        Tensor forward(Tensor input_state, string target) {
            if(target=="main"){
                return main_net_->forward(input_state);
            }
            else{
                return target_net_->forward(input_state);
            }
    }   
        void update_target_network() {
            auto q_params = main_net_->parameters();
            auto target_params = target_net_->parameters();
            for (size_t i = 0; i < target_params.size(); ++i) {
                target_params[i].data().mul_(1 - tau_);
                target_params[i].data().add_(q_params[i].data() * tau_);
                //cout << q_params[i].data() << endl;
                }   
                
    }

        int optimize_net(int batch_size, float gamma_){
        
            if (replay_memory.size()<batch_size){
                return 0;
            }
                std::vector<Tensor> state_batch;
                std::vector<int> action_batch;
                std::vector<float> reward_batch;
                std::vector<Tensor> next_state_batch;
                std::vector<int> done_batch;

              //sample batch of batch_size size
            for (int i = 0; i < batch_size; i++) {
                // Get experience from replay memory
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> dist(0, replay_memory.size() - 1);
                // Get a random index within the range of valid indices
                int random_index = dist(gen);
                auto experience = replay_memory[random_index];
                torch::Tensor state;
                torch::Tensor next_state;
                int action;
                float reward;
                int done;
                // Append experience to corresponding batch vector
                state_batch.push_back(std::get<0>(experience));
                action_batch.push_back(std::get<1>(experience));
                reward_batch.push_back(std::get<2>(experience));
                next_state_batch.push_back(std::get<3>(experience));
                done_batch.push_back(std::get<4>(experience));
    }
             // Convert batch tensors to a single tensor for each batch
            auto state_tensor = torch::stack(state_batch, 0).to(kFloat);
            auto next_state_tensor = torch::stack(next_state_batch, 0).to(kFloat);
            auto action_tensor = torch::from_blob(action_batch.data(), {batch_size, 1}, torch::kInt).to(device_);
            auto reward_tensor = torch::from_blob(reward_batch.data(), {batch_size, 1}, torch::kFloat32).to(device_);
            auto done_tensor = torch::from_blob(done_batch.data(), {batch_size, 1}, torch::kInt).to(device_);

            auto action_tensor_int64 = action_tensor.to(torch::kLong);

            //a mask might be necessary here to filter out all None states from the batches

            auto q_values = forward(state_tensor, "main").gather(1,action_tensor_int64);
            auto next_q_values=forward(next_state_tensor, "target").gather(1,action_tensor_int64);

            auto target_q_values = reward_tensor + (gamma_ * next_q_values) * (1 - done_tensor);
            //auto target_q_values = reward_tensor + (gamma_ * next_q_values.unsqueeze(1).squeeze(1)) * (1 - done_tensor);
            auto loss = torch::smooth_l1_loss(q_values, target_q_values.detach());
            
            optimizer->zero_grad();
            loss.backward();
            optimizer->step();

 
            update_target_network();

            return 0;
        
        }
        
        
        
        
        
        
        int select_action(Tensor state){

            Tensor state_float = state.to(torch::kFloat32);
            Tensor main_q_values = forward(state_float, "main");
            int action;
            epsilon_threshold_ = epsilon_end_ + (epsilon_start_ - epsilon_end_) * std::exp(-1. * steps_done / epsilon_decay_);
            steps_done += 1;
            //epsilon is the probability to explore the action space instead of exploit by generating a random value of 0 or 1
            //and comparing it to epsilon, if this is false then we go over to exploitation
            
            if (torch::rand({1}).item<float>() < epsilon_threshold_) {
            // Sample a random index from the action space
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> dist(0, env.get_action_space() - 1);
                int index = dist(gen);
        
        // Select the action corresponding to the sampled index
                action = index;
            } 
            else {
        // Select the action with the highest Q-value
            action = main_q_values.argmax().item<int>();
            }
            
        return action;


        }
        void train(int max_episodes, int max_steps, float gamma, int batch_size, int replay_memory_size){
            

        for (int episode=1; episode<= max_episodes; episode++){
            Tensor state= env.reset();
            float episode_reward = 0.0f;
            bool done=false;

            while(!done){
                int action = select_action(state);

                Tensor observation;
                float reward;

                // Take action and observe next state, reward, and done
                std::tie(observation, reward, done) = env.step(action);
                
                //maybe set next_state to None when done=true
                
                if (done){
                    //std::shared_ptr<torch::Tensor> next_state_ptr = std::make_shared<torch::Tensor>(std::move(observation));
                    //next_state_ptr = nullptr;
                }
                //might need to check for correct types here: Tensor<float>
                auto experience = std::make_tuple(state.clone(), action, reward, observation.clone(), done);

                replay_memory.push_back(experience);
                episode_reward += reward;
                
                state = observation;

                optimize_net(batch_size, gamma);


                if(done){

                cout << "Episode: " << episode << "  Reward: "<< episode_reward <<  "  Epsilon: "<< epsilon_threshold_ <<endl;
                //plot rewards
                }


        }

    }


        }

private:
    int state_size_;
    int action_size_;
    int hidden_size_;
    float learning_rate_;
    int steps_done=0;
    double epsilon_start_;
    double epsilon_end_;
    double epsilon_decay_;
    double epsilon_threshold_;
    double tau_;
    deque<tuple<Tensor, int, float, Tensor, int>> replay_memory;

    CartPoleEnv env;
    Device device_;
    nn::Sequential main_net_;
    unique_ptr<torch::optim::Optimizer> optimizer;
    nn::Sequential target_net_; 
         
};






int main(){
//torch::manual_seed(0);
    float learning_rate= 0.0001;
    int num_episodes=5000;
    int max_steps=5000;
    float gamma=0.99;
    int batch_size= 128;
    int replay_memory_size= 10000;
    float epsilon_start=0.9;
    float epsilon_end=0.005;
    float epsilon_decay=1500;
    float tau=0.01;

    DQNAgent agent(learning_rate,epsilon_start, epsilon_end, epsilon_decay, tau);

    agent.train(num_episodes, max_steps, gamma, batch_size, replay_memory_size);



}

