#include "cartpole.hpp"
#include "dqn.hpp"

int main() {
    // Define the hyperparameters
    int32_t num_episodes = 1000;
    int32_t max_timesteps = 1000;
    float gamma = 0.99;
    float epsilon_start = 1.0;
    float epsilon_end = 0.01;
    float epsilon_decay = 0.995;
    int32_t batch_size = 64;
    int32_t memory_capacity = 10000;
    float lr = 0.001;
    int32_t target_update_frequency = 10;

    // Create the CartPole environment
    CartPole env;

    // Create the DQN agent
    DQN agent(env.get_observation_space().shape(), env.get_action_space().n, lr, gamma, epsilon_start, epsilon_end, epsilon_decay, batch_size, memory_capacity, target_update_frequency);

    // Train the agent for the specified number of episodes
    for (int32_t i = 0; i < num_episodes; i++) {
        // Reset the environment and get the initial state
        auto state = env.reset();

        // Run the episode for a maximum of max_timesteps
        for (int32_t t = 0; t < max_timesteps; t++) {
            // Select an action using the agent's policy
            int32_t action = agent.select_action(state);

            // Take the selected action and get the next state, reward, and done flag
            auto [next_state, reward, done] = env.step(action);

            // Store the experience in the agent's replay buffer
            agent.store_experience(state, action, reward, next_state, done);

            // Update the agent's Q-network by sampling experiences from the replay buffer
            agent.update_q_network();

            // Update the target network if necessary
            agent.update_target_network();

            // Update the state
            state = next_state;

            // Break the loop if the episode is done
            if (done) {
                break;
            }
        }

        // Decay the exploration rate after each episode
        agent.decay_epsilon();
    }

    return 0;
}
