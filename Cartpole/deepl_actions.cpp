#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <memory>
#include <queue>
#include "CartpoleEnv2.h"
#include <cstdint>

using namespace torch;
//using namespace std;

// DQN Agent
class DQNAgent {
public:
//network parameters
    //torch::optim::Adam optimizer_;  
    DQNAgent(int state_size,int action_size,int hidden_size, float learning_rate)
    //after : class vars are being filled either with input or with other values like "cpu"
        : state_size_(state_size), action_size_(action_size), hidden_size_(hidden_size), 
        learning_rate_(learning_rate), device_("cpu"), 
          q_network(nn::Sequential(
            nn::Linear(nn::LinearOptions(state_size, hidden_size).bias(false)),
            nn::Functional(torch::relu),
            nn::Linear(nn::LinearOptions(hidden_size, action_size).bias(false))
        )),
        optimizer_(q_network->parameters(), torch::optim::AdamOptions(learning_rate_))
        {
        //constructor
        // Initialize Q-network
      
        //    torch::nn::Linear(num_states, hidden_size),
        //    torch::nn::ReLU(),
        //    torch::nn::Linear(hidden_size, hidden_size),
        //    torch::nn::ReLU(),
        //    torch::nn::Linear(hidden_size, num_actions)
        // Initialize target Q-network
        // Create a deep copy of the Q-network for the target network
         q_network->to(device_);
        //torch::optim::Adam optimizer(q_network->parameters(), torch::optim::AdamOptions(learning_rate));
        //might be necessary to create as private var and adress in the update_q_network function instead of tthe q_network
        //auto target_q_network_ = std::make_shared<torch::nn::Sequential>(*q_network);
        //torch::nn::Sequential target_q_network_; 
        //std::shared_ptr<torch::nn::Sequential> q_net_copy = std::make_shared<torch::nn::Sequential>(q_network);
        //target_q_network_ = *q_net_copy;
        //target_q_network_->to(device_);

        // Define the optimizer

    }
    //constructor Ende
    torch::optim::Adam optimizer_;
     
    //class methods
    int  select_action(torch::Tensor state, float epsilon, CartPoleEnv env);
    void train(CartPoleEnv& env, int num_episodes, int max_steps, float gamma, float epsilon, int batch_size,
         int replay_memory_size, float epsilon_decay, float epsilon_end, int target_update_frequency_)
    //other possible hyperparameters: learning rate(specified manually in optimizer), target_update_frequency(f.e. 10))
    {
        std::deque<std::tuple<Tensor, int, float, Tensor, int>> replay_memory;
        for (int episode = 1; episode <= num_episodes; episode++) {
            // Reset environment
            env.reset();

            // Reset episode statistics
            float episode_reward = 0.0f;
            int episode_steps = 0;

            // Run episode
            while (episode_steps < max_steps) {
                //std::vector<float> state = env.get_observation_space().to(device_);
                // Get the observation space from the environment
                std::vector<float> obs = env.get_observation_space();

                // Convert the observation space to a torch::Tensor
                auto state = torch::from_blob(obs.data(), {1, static_cast<long>(obs.size())}).to(device_);
                // You can now use the obs_tensor as input to your neural network

                // Choose action
                int action = select_action(state, epsilon, env);
                std::vector<float> next_state;
                float reward;
                int done;
                // Take action and observe next state, reward, and done
                std::tie(next_state, reward, done) = env.step(action);
                torch::TensorOptions options(torch::kCPU); // or torch::kCUDA if you want to use GPU
                options = options.dtype(torch::kFloat32);
                //auto [next_state, reward, done] = env.step(action);
                torch::Tensor next_state_tensor = torch::from_blob(next_state.data(), {1, static_cast<int64_t>(next_state.size())}, options);

                // Move the tensor to the device
                next_state_tensor = next_state_tensor.to(device_);
                //next_state = next_state.to(device_);
                //Tensor reward_tensor = from_blob(&reward, {1}, TensorOptions().dtype(kFloat32)).to(device_);

                // Store experience in replay memory
                auto experience = std::make_tuple(state, action, reward, next_state_tensor, done);
                replay_memory.push_back(experience);
                if (replay_memory.size() > replay_memory_size) {
                    //assures that the replay_memory is only of the size of the replay_memory size
                    replay_memory.pop_front();
                }

                // Update episode statistics
                episode_reward += reward;
                episode_steps++;

                // Update Q-network
                if (replay_memory.size() >= batch_size) {
                    update_q_network(gamma, batch_size, replay_memory);
                }
                // Update target Q-network
                if (episode_steps % target_update_frequency_ == 0) { 
                    // torch::nn::Sequential target_q_network_; 
                    //std::shared_ptr<torch::nn::Sequential> q_net_copy = std::make_shared<torch::nn::Sequential>(q_network);
                    //target_q_network_ = *q_net_copy;   
                    //auto target_q_network_ = std::make_shared<torch::nn::Sequential>(*q_network);
                    //target_q_network_ = nn::clone(q_network_);
                 
                }
                //maybe go into eval mode with target_q_network

                // Check if episode is done
                if (done==1) {
                    break;
                }

                // Update current state
                //state = next_state.clone();
                state= next_state_tensor.clone();
            }

            // Print episode statistics
            std::cout << "Episode " << episode << " - Reward: " << episode_reward << " - Steps: " << episode_steps << std::endl;
           // decay_epsilon(epsilon,epsilon_decay, epsilon_end);
        }
    }
        //void update();
        //following methods dont need to be initializied here, because there are inside of the class declaration,
        //in contrast to "select action"
        //void decay_epsilon(float epsilon_, float epsilon_decay_, float epsilon_end_);
        //void train(int num_episodes, int max_steps);
    //private class vars
    private:
        int state_size_;
        int action_size_;
        int hidden_size_;
        float learning_rate_;
        torch::Device device_;
        torch::nn::Sequential q_network; 
       //torch::optim::Adam optimizer_;

//
//        void decay_epsilon(float epsilon_, float epsilon_decay_, float epsilon_end_) {
//    if (epsilon_ > epsilon_end_) {
//        epsilon_ *= epsilon_decay_;
//        epsilon_ = std::max(epsilon_, epsilon_end_);
//    }
//}

void update_q_network(float gamma, int batch_size, std::deque<std::tuple<Tensor, int, float, Tensor, int>>& replay_memory) {
    // Create batch tensors
    std::vector<Tensor> state_batch;
    std::vector<int> action_batch;
    std::vector<float> reward_batch;
    std::vector<Tensor> next_state_batch;
    std::vector<int> done_batch;
    for (int i = 0; i < batch_size; i++) {
        // Get experience from replay memory
        auto experience = replay_memory[rand() % replay_memory.size()];

        // Append experience to corresponding batch vector
        state_batch.push_back(std::get<0>(experience));
        action_batch.push_back(std::get<1>(experience));
        reward_batch.push_back(std::get<2>(experience));
        next_state_batch.push_back(std::get<3>(experience));
        done_batch.push_back(std::get<4>(experience));
    }

    // Convert batch tensors to a single tensor for each batch
    auto state_tensor = torch::cat(state_batch, 0);
    auto action_tensor = torch::from_blob(action_batch.data(), {batch_size, 1}, torch::kInt).to(device_);
    auto reward_tensor = torch::from_blob(reward_batch.data(), {batch_size, 1}, torch::kFloat32).to(device_);
    auto next_state_tensor = torch::cat(next_state_batch, 0);
    auto testing =torch::from_blob(action_batch.data(), {batch_size, 1}, torch::kInt);
    auto done_tensor = torch::from_blob(done_batch.data(), {batch_size, 1}, torch::kInt).to(device_);
//    auto done_tensor = torch::from_blob(done_batch.data(), {batch_size, 1}, torch::kBool);
    // Compute Q-values for state-action pairs
    auto q_values = q_network->forward(state_tensor).gather(1, action_tensor);

    // Compute target Q-values for next state-action pairs
    auto next_q_values = torch::zeros_like(q_values);
    auto indices = std::get<1>(q_values.max(1, true));

    //next_q_values.masked_scatter_(done_tensor.logical_not(), q_network->forward(next_state_tensor).detach().max(1).indices.unsqueeze(1));
    auto target_q_values = reward_tensor + gamma * next_q_values;

    // Compute loss and backpropagate
    auto loss = torch::mse_loss(q_values, target_q_values);
    optimizer_.zero_grad();
    loss.backward();
    optimizer_.step();
    
}

};
 //int DQNAgent::select_action(Tensor state, float epsilon, CartPoleEnv env ) {
 //   Tensor q_values = q_network->forward(state);
 //   int action;
 //   //epsilon is the probability to explore the action space instead of exploit by generating a random value of 0 or 1
 //   //and comparing it to epsilon, if this is false then we go over to exploitation
 //   if (torch::rand({1}).item<float>() < epsilon) {
 //       // Sample a random index from the action space
 //       std::random_device rd;
 //       std::mt19937 gen(rd());
 //       std::uniform_int_distribution<> dist(0, env.get_action_space() - 1);
 //       int index = dist(gen);
 //       cout << "EPSILON: " << endl;
 //       cout << epsilon << endl;
 //       // Select the action corresponding to the sampled index
 //       action = env.get_action_space();
 //   } else {
 //       // Select the action with the highest Q-value
 //       action = q_values.argmax(1).item<int>();
 //   }
 //   return action;
 //}


int main(){
    //int int int float, state_size,action_size,hidden_size,learning_rate
    CartPoleEnv env;

    int state_size = env.get_state().size();
    int action_size = env.get_action_space();
    float hidden_size= 16;
    float learning_rate= 0.01;
    int num_episodes=1000;
    int max_steps=200;
    //importance of future rewards => closer o 1: agent will consider future rewards more important than immediate rewards 
    //=> long term focused policy, if closer to 0 => more focused on immediate rewards
    float gamma=0.99;
    float epsilon=0.15;
    int batch_size= 50;
    int replay_memory_size= 10000;
    //agent will explore less and exploit more earlier in training, 
    //while a lower value means that the agent will explore more and exploit less earlier in training.
    float epsilon_decay=0.995;
    float epsilon_end=0.01;
    DQNAgent agent(state_size,action_size,hidden_size,learning_rate);
    int target_update_frequency=100;
//CartPoleEnv &env, int num_episodes, int max_steps, float gamma, float epsilon, int batch_size, int replay_memory_size,
//float epsilon_decay, float epsilon_end, int target_update_frequency_, torch::optim::Adam optimizer
    agent.train(env, num_episodes, max_steps, gamma, epsilon, batch_size, replay_memory_size, 
                epsilon_decay, epsilon_end, target_update_frequency);
    // Train agent on environment
    //agent.train(env, 1000);
}