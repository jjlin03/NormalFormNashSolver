import jax
import jax.numpy as jnp
import optax
import haiku as hk
from typing import NamedTuple
from random import shuffle

class TrainingState(NamedTuple):
  params: hk.Params
  opt_state: optax.OptState

class Model:
    def __init__(self, network_fn):
        """
        The Model class approximates Nash equilibrium in Normal-Form games using neural networks.

        Parameters:
        - network_fn: The network function used to train and test the model.
        """
        self.network = hk.without_apply_rng(hk.transform(network_fn))
        self.optimizer = None
        self.state = None

    def calculate_utilities(self, payoff_tensor, actions):
        """
        Calculates utility of each action for each player

        Parameters:
        - payoff_tensor: Utility tensor of shape (num_batches, num_players, num_actions, ..., num_actions)
        - actions: Probability distribution over actions for each player (num_batches, num_players, num_actions)

        Returns:
        - List of utility functions for each player [(num_batches, num_actions)] * num_players
        """
        num_players = jnp.shape(payoff_tensor)[1]
        player_utilities = []

        for a in range(num_players):
            player_utilities.append(jnp.copy(payoff_tensor[:,a]))
            for b in range(num_players - 1, -1, -1):
                if b == a:
                    # Moves player of interest in front of batch dim
                    player_utilities[-1] = jnp.moveaxis(player_utilities[-1], -1, 1)
                else:
                    player_utilities[-1] = jnp.einsum('i...jk,ik->i...j', player_utilities[-1], actions[:,b])

        return player_utilities
    
    def calculate_regret(self, player_utilities, actions):
        """
        Calculates average exploitability for the game

        Parameters:
        - player_utilities: List of utility functions for each player [(num_batches, num_actions)] * num_players
        - actions: Probability distribution over actions for each player (num_batches, num_players, num_actions)

        Return:
        - Exploitability averaged over each player
        """
        num_players = len(player_utilities)
        num_batches = jnp.shape(actions)[0]

        regret_sum = jnp.zeros(num_batches)
        for a in range(num_players):
            curr_utility = jnp.einsum('ij,ij->i', player_utilities[a], actions[:,a])
            curr_regret = jnp.max(player_utilities[a], axis=1) - curr_utility 
            regret_sum += curr_regret

        regret = regret_sum / num_players
        regret = jnp.mean(regret)
        return regret
    
    @jax.jit
    def loss_fn(self, params, payoff_tensor):
        """
        Loss function of neural network model.

        Parameters:
        - params: Neural network weights from network function
        - payoff_tensor: Utility tensor of shape (num_batches, num_players, num_actions, ..., num_actions)

        Return:
        - Exploitability averaged over each player
        """
        actions = self.network.apply(params, payoff_tensor)
        player_utilities = self.calculate_utilities(payoff_tensor, actions)
        loss = self.calculate_regret(player_utilities, actions)
        return loss
    
    @jax.jit
    def update(self, state: TrainingState, batch) -> TrainingState:
        """
        Performs gradient descent over each batch.

        Parameters:
        - state: Current TrainingState to be updated
        - batch: New payoff tensor

        Returns:
        - Updated TrainingState
        """
        grads = jax.grad(self.loss_fn)(state.params, batch)
        updates, opt_state = self.optimizer.update(grads, state.opt_state)
        params = optax.apply_updates(state.params, updates)
        return TrainingState(params, opt_state)

    def train(self, data, epochs, optimizer=optax.adam(1e-3)):
        """
        Trains the model. Updates the TrainingState of the model at the end of training.
        This function has to be ran before test() can be called.

        Parameters:
        - data: List of utility tensors of shape (num_batches, num_players, num_actions, ..., num_actions)
            - List is assumed to be non-empty
        - epochs: Number of epochs the model will train for
        - optimizer: Optimizer from optax class to be used in model
        """
        self.optimizer = optimizer
        initial_params = self.network.init(jax.random.PRNGKey(seed=0), data[0])
        initial_opt_state = self.optimizer.init(initial_params)
        state = TrainingState(initial_params, initial_opt_state)

        loss = 0
        for i in range(epochs):
            for j in range(1, len(data)):
                state = self.update(state, data[j])
                loss += self.loss_fn(state.params, data[j])
            print(f'Completed Epoch {i+1}/epochs of training.')
            print(f'Average epoch loss of {loss/len(data)}')
            loss = 0
            shuffle(data)

        self.state = state

    def test(self, utilities):
        """
        Approximates the Nash Equilibirum of a Normal Form Game.

        Parameters:
        - utilities: Game utility tensor of shape (num_players, num_actions, ..., num_actions)
        
        Returns:
        - Probability distribution over actions for each player
            - shape=(num_players, num_actions)
        """
        payoff_tensor = jnp.expand_dims(utilities, axis=0)
        loss = self.loss_fn(self.state.params, payoff_tensor)
        print(f'Total exploitability is {loss}')
        actions = self.network.apply(self.state.params, payoff_tensor)
        actions = jnp.squeeze(actions, axis=0)

        return actions