#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

// Define the neural network
struct QNetwork : torch::nn::Module {
    QNetwork(int64_t num_inputs, int64_t num_actions) {
        fc1 = register_module("fc1", torch::nn::Linear(num_inputs, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 64));
        fc3 = register_module("fc3", torch::nn::Linear(64, num_actions));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1(x));
        x = torch::relu(fc2(x));
        x = fc3(x);
        return x;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

// Define the agent
class Agent {
public:
    Agent(int64_t num_inputs, int64_t num_actions, double gamma, double epsilon, double epsilon_min, double epsilon_decay, double lr) :
        gamma(gamma), epsilon(epsilon), epsilon_min(epsilon_min), epsilon_decay(epsilon_decay), q_network(num_inputs, num_actions), optimizer(q_network->parameters(), torch::optim::AdamOptions(lr)) {}

    int64_t select_action(torch::Tensor state) {
        // Epsilon-greedy action selection
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(rng) < epsilon) {
            return dist_action(rng);
        } else {
            torch::NoGradGuard no_grad;
            auto q_values = q_network->forward(state);
            auto max_q = q_values.max(0);
            return max_q.indices().item<int64_t>();
        }
    }

    void update_replay_buffer(torch::Tensor state, int64_t action, double reward, torch::Tensor next_state, bool done) {
        replay_buffer.push_back(std::make_tuple(state, action, reward, next_state, done));
        if (replay_buffer.size() > replay_buffer_size) {
            replay_buffer.pop_front();
        }
    }

    void train() {
        if (replay_buffer.size() < batch_size) {
            return;
        }

        // Sample a batch of experiences from the replay buffer
        std::vector<int64_t> indices(batch_size);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        std::vector<torch::Tensor> states, next_states;
        std::vector<int64_t> actions;
        std::vector<double> rewards;
        std::vector<bool> dones;

        for (int i = 0; i < batch_size; i++) {
            auto [state, action, reward, next_state, done] = replay_buffer[indices[i]];
            states.push_back(state);
            actions.push_back(action);
            rewards.push_back(reward);
            next_states.push_back(next_state);
            dones.push_back(done);
        }

        auto states_tensor = torch::stack(states);
        auto next_states_tensor = torch::stack(next_states);
        auto actions_tensor = torch::from_blob(actions.data(), {batch_size}, torch::kInt64);
        auto rewards_tensor = torch::from_blob(rewards.data(), {batch_size}, torch::kDouble);
        auto dones_tensor = torch::from_blob(dones.data(), {batch_size}, torch::kBool);

        // Compute the Q-values for the current state and the next state
        auto q_values = q_network->forward(states_tensor);
        auto next_q_values = q_network->forward(next_states_tensor);

        // Compute the target Q-values using the Bellman equation
        auto target_q_values = rewards_tensor + gamma * next_q_values.max(1).values() * (1 - dones_tensor);

        // Compute the loss between the predicted Q-values and the target Q-values
        auto loss = torch::nn::functional::mse_loss(q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1), target_q_values.detach());

        // Update the weights of the neural network using backpropagation
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        // Decay the exploration rate
        if (epsilon > epsilon_min) {
            epsilon *= epsilon_decay;
        }
    }

private:
    double gamma;
    double epsilon;
    double epsilon_min;
    double epsilon_decay;
    double lr;
    int64_t batch_size = 64;
    int64_t replay_buffer_size = 10000;
    std::deque<std::tuple<torch::Tensor, int64_t, double, torch::Tensor, bool>> replay_buffer;
    std::unique_ptr<QNetwork> q_network = nullptr;
    torch::optim::Adam optimizer;
    std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<int64_t> dist_action{0, 1};
};

// Define the main function
int main() {
    // Set up the environment
    int64_t num_inputs = 4;
    int64_t num_actions = 2;
    double gamma = 0.99;
    double epsilon = 1.0;
    double epsilon_min = 0.01;
    double epsilon_decay = 0.995;
    double lr = 0.001;
    int64_t num_episodes = 1000;
    int64_t max_steps = 1000;

    // Create the agent
    Agent agent(num_inputs, num_actions, gamma, epsilon, epsilon_min, epsilon_decay, lr);

    // Train the agent
    for (int64_t episode = 0; episode < num_episodes; episode++) {
        // Reset the environment
        torch::Tensor state = torch::zeros({num_inputs});
        double total_reward = 0.0;
        bool done = false;

        for (int64_t step = 0; step < max_steps && !done; step++) {
            // Select an action
            int64_t action = agent.select_action(state);

            // Take the action and observe the next state and reward
            torch::Tensor next_state = torch::zeros({num_inputs});
            double reward = 0.0;
            bool done = false;

            // Update the replay buffer
            agent.update_replay_buffer(state, action, reward, next_state, done);

            // Train the agent
            agent.train();

            // Update the state and total reward
            state = next_state;
            total_reward += reward;
        }

        // Print the total reward for the episode
        std::cout << "Episode " << episode << " Total Reward: " << total_reward << std::endl;
    }

    return 0;
}
