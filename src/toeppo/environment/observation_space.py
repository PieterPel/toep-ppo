import gym
from gym.spaces import Dict, Discrete, MultiDiscrete


class ToepObservationSpace(gym.Space):
    def __init__(
        self,
        num_players,
        num_cards_per_player,
        card_to_number_mapping: dict,
        max_score=15,
    ):
        self.num_players = num_players
        self.max_cards_per_pile = num_cards_per_player
        self.card_to_number_dict = card_to_number_mapping
        highest_number = max(self.card_to_number_dict.values())

        # Define subspaces for different elements of the observation
        self.player_hands_space = MultiDiscrete(
            [highest_number] * num_cards_per_player
        )
        self.player_piles_space = MultiDiscrete(
            [highest_number] * num_cards_per_player
        )
        self.player_scores_space = (
            MultiDiscrete([max_score] * num_players),
        )  # TODO: maybe change 15 to some kind of variable
        self.turn_number_space = Discrete(num_players)
        self.sub_round_number_space = Discrete(num_cards_per_player)

        # Combine all subspaces into a dictionary space
        self.observation_space = Dict(
            {
                "player_hands": self.player_hands_space,
                "player_piles": self.player_piles_space,
                "player_scores": self.player_scores_space,
                "turn_number": self.turn_number_space,
                "sub_round_number": self.sub_round_number_space,
            }
        )

    def sample(self):
        # Sample a random observation from each subspace
        return {
            "player_hands": self.player_hands_space.sample(),
            "player_piles": self.player_piles_space.sample(),
            "player_scores": [
                self.player_scores_space.sample()
                for _ in range(self.num_players)
            ],
            "turn_number": self.turn_number_space.sample(),
            "sub_round_number": self.sub_round_number_space.sample(),
        }

    def contains(self, x):
        # Check if x is a valid observation
        if not isinstance(x, dict):
            return False
        if set(x.keys()) != set(
            [
                "player_hands",
                "player_piles",
                "player_scores",
                "turn_number",
                "sub_round_number",
            ]
        ):
            return False
        if not self.player_hands_space.contains(x["player_hands"]):
            return False
        if not self.player_piles_space.contains(x["player_piles"]):
            return False
        if not self.player_scores_space.contains(x["player_scores"]):
            return False
        if not self.turn_number_space.contains(x["turn_number"]):
            return False
        if not self.sub_round_number_space.contains(x["sub_round_number"]):
            return False
        return True
