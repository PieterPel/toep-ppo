import gymnasium as gym
from gymnasium.spaces import (
    Dict,
    Discrete,
    MultiDiscrete,
    MultiBinary,
    Box,
    flatten,
    flatten_space,
)
import numpy as np


class ToepObservationSpace:
    def __init__(
        self,
        num_players,
        num_cards_per_player,
        card_to_number_mapping: dict,
        max_score=15,
        max_score_multiplier=10,
    ):
        self.num_players = num_players
        self.max_cards_per_pile = num_cards_per_player
        self.card_to_number_dict = card_to_number_mapping
        self.num_cards_per_player = num_cards_per_player
        highest_number = max(self.card_to_number_dict.values())

        # Define subspaces for different elements of the observation
        self.player_hands_space = MultiDiscrete(
            [highest_number + 1]
            * self.num_cards_per_player
            * self.num_players,
            dtype=np.int32,
        )
        self.player_piles_space = MultiDiscrete(
            [highest_number + 1]
            * self.num_cards_per_player
            * self.num_players,
            dtype=np.int32,
        )
        self.player_scores_space = MultiDiscrete(
            [max_score * max_score_multiplier] * num_players, dtype=np.int32
        )
        self.turn_number_space = Discrete(num_players + 1)
        self.sub_round_number_space = Discrete(self.num_cards_per_player + 1)
        self.action_type_space = Discrete(4)

        # Combine all subspaces into a dictionary space
        self.observation_space_dict = Dict(
            {
                "player_hands": self.player_hands_space,
                "player_piles": self.player_piles_space,
                "player_scores": self.player_scores_space,
                "turn_number": self.turn_number_space,
                "sub_round_number": self.sub_round_number_space,
                "action_type": self.action_type_space,
            }
        )

        # Create a Box space
        self.observation_space_flattened = flatten_space(
            self.observation_space_dict
        )

        self.observation_space = Dict(
            {
                "observation": self.observation_space_flattened,
                "action_mask": MultiBinary(39),
            }
        )

    def empty_space(self):
        # Create empty observation space for player hands and piles
        player_hands_empty = np.zeros(
            (self.num_players, self.num_cards_per_player * self.num_players),
            dtype=int,
        )
        player_piles_empty = np.zeros(
            (self.num_players, self.num_cards_per_player * self.num_players),
            dtype=int,
        )

        # Create empty observation space for player scores
        player_scores_empty = np.zeros((self.num_players,), dtype=int)

        # Set other values to zero
        turn_number_empty = 0
        sub_round_number_empty = 0
        action_type = 0

        dictionary = {
            "player_hands": player_hands_empty,
            "player_piles": player_piles_empty,
            "player_scores": player_scores_empty,
            "turn_number": turn_number_empty,
            "sub_round_number": sub_round_number_empty,
            "action_type": action_type,
        }

        # return flatten(self.observation_space_dict, dictionary)
        return dictionary

    @property
    def shape(self):
        # Calculate the shape of each subspace
        player_hands_shape = self.player_hands_space.shape
        player_piles_shape = self.player_piles_space.shape
        player_scores_shape = self.player_scores_space.shape
        turn_number_shape = self.turn_number_space.shape
        sub_round_number_shape = self.sub_round_number_space.shape
        action_type_shape = self.action_type_space.shape

        # Calculate the total size of the observation space
        total_size = int(
            np.prod(player_hands_shape)
            + np.prod(player_piles_shape)
            + np.prod(player_scores_shape)
            + np.prod(turn_number_shape)
            + np.prod(sub_round_number_shape)
            + np.prod(action_type_shape)
        )

        return (total_size,)
