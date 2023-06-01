To make a deep copy of a torch nn:
  std::shared_ptr<torch::nn::Sequential> new_nn_ref = std::make_shared<torch::nn::Sequential>(old_nn);

To access the normal Sequential model, dereferencing is necessary:
 torch::nn::Sequential& new_nn_deref = *new_nn_ref;

No normal Sequential model function can be applied!


Segmentation fault needs to be adressed => solved

batch doesnt store experience of state properly => solved

the q_values are nan after some time beeing normal!
->update_q_network function has a lot of speculation => build knowledge => problem is definitely update_q_network function
=> error is not in replay buffer to batch function, states look good but problem might e in other parameters
Before q_values are calculated, some gradients are nan => why? => check other parameters than state for anomalies
=> weights are -nan first => maybe numerical instability
=>Gradients after loss.backwards() function have nan first => propagates through weights
=>loss seems to be nan before that
=> target_q_values has nan value => reason for loss value of nan
=> mutliple nan in next_q_values => reason for nan in target_q_value
=> Infinte value in next_state_tensor => leads to nan in next_q_value
=> problem in next state batch, likely far to high values here compared to state batch
=>problem was located in replay buffer, error value can either be inf or nan
=> I suspect error in set() function so probably in environment
apparently the replay_memory.push_back(experience) function alters the next_state in experience significantly -> no idea why
=> it was next_state that would not do a deep copy but a reference
=> error solved!
=> next problem: model doesnt get better over time -> adjust parameters and reward function
=> maybe look at if the cart or the falling pole is the problem
=> visualisieren des problems
=> try probkem with simple q-learning ->bookmarks
=> model converged once pretty efficiently after around 1000-1500 Episodes up to consistent 100-200 reward (Epsilon could be reason)

Carful with the target_network update, it is significant and not very easy in c++. orientate on the python version

target_net=main_net-> copy() need to be investigated