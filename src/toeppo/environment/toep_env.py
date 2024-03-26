from pettingzoo import AECEnv
import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete, flatten
from gymnasium.wrappers.flatten_observation import FlattenObservation
from .toep_game import ToepGame, Player, ActionType, CARDS_PER_PLAYER
from .observation_space import ToepObservationSpace
from pettingzoo.utils import agent_selector, wrappers
import functools
import numpy as np
import copy
import logging


def env(**kwargs):
    return ToepEnv.env(n_players=4)


class ToepEnv(AECEnv):
    ACTION_SPACE_SIZE = 39
    metadata = {
        "is_parallelizable": True,
        "name": "toeppo",
    }

    def __init__(
        self, n_players, losing_penalty_multiplier=10, render_mode=None
    ):
        # self.n_players = n_players
        self.n_players = 4
        self.losing_penalty_multiplier = losing_penalty_multiplier
        self.logger = logging.getLogger(__name__)

        # Create the game where we will operate in
        self.game = ToepGame(self.n_players)

        self.number_to_card_dict = {
            number: card for number, card in zip(range(7, 39), self.game.deck)
        }  # NOTE: deck maybe isnt iterable automatically?

        self.card_to_number_dict = {
            card: number for number, card in self.number_to_card_dict.items()
        }

        self.observation_space_base = ToepObservationSpace(
            self.n_players, CARDS_PER_PLAYER, self.card_to_number_dict
        )

        self.possible_agents = [
            "player_" + str(r) for r in range(1, self.n_players + 1)
        ]

        self.action_spaces = {
            agent: self.action_space(agent) for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: self.observation_space(agent)
            for agent in self.possible_agents
        }
        self.render_mode = render_mode

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):

        return self.observation_space_base.observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # 0: toep
        # 1: go
        # 2: fold
        # 3: call vuile was
        # 4: do not call vuile was
        # 5: check
        # 6: trust
        # 7 - 38 play a certain card

        return Discrete(self.ACTION_SPACE_SIZE)

    def reset(self, *, seed=None, options=None):

        self.logger.info("Resetting the environment")

        self.game.reset()

        self.previous_scores_dict = self.get_current_scores()

        self.agents = self.possible_agents[:]
        self.agent_to_player_dict = {
            agent: player
            for agent, player in zip(self.agents, self.game.players)
        }

        self.player_to_agent_dict = {
            player: agent
            for agent, player in self.agent_to_player_dict.items()
        }

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        # self.state = {agent: NONE for agent in self.agents} # NOTE: not sure what this did in the original code

        self.num_moves = (
            0  # NOTE: should probably be replaced with somehting else
        )

        # Start the first round
        first_player, first_action = self.game.start_round()

        self.observations = self.get_observations(first_action)
        self.infos = self.get_infos(first_player, first_action)

        self.agent_selection = self.player_to_agent_dict[first_player]

        self.logger.info(f"First action: {first_action}")

        return self.observations, self.infos

    def step(self, action: ActionType):
        # NOTE: orginal code first check for terminations or truncations, not sure if that is necessary here since all players are done at the same time always
        self.num_moves += 1
        agent = self.agent_selection

        self.logger.info(f"Taking a step: action {action} of {agent}")

        # NOTE: I do not understand this
        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # if self.game.ended_game:
        #     self.logger.info("The game has ended")

        #     # self.agents = {}

        #     # self.rewards = {agent: 0 for agent in self.possible_agents}
        #     # self._accumulate_rewards()

        #     return (
        #         self.observations,
        #         self.rewards,
        #         self.truncations,
        #         self.terminations,
        #         self.infos,
        #     )

        # Check for legal action
        if self.invalid_action(action):
            self.logger.info(f"{action} of {agent} is invalid!")

            self.rewards = self.get_rewards()
            self.rewards[agent] = -1 * self.invalid_action_penalty(
                action, self.action_type
            )
            self._accumulate_rewards()

            player = self.agent_to_player_dict[agent]
            self.infos = self.get_infos(
                player, self.action_type, extra_mask=action
            )

            # Try it again
            # return (
            #     self.observations,
            #     self.rewards,
            #     self.truncations,
            #     self.terminations,
            #     self.infos,
            # )

        # Convert action to action for player
        player = self.agent_to_player_dict[agent]

        self.logger.info(f"{player} took action {action}")

        next_player, self.action_type = self.handle_action_for_player(
            player, action
        )

        # if self.game.ended_game:
        #     self.rewards = self.get_rewards()

        # # NOTE this is very ugly but it works
        # for player in self.game.losing_players:
        #     agent_ = self.player_to_agent_dict[player]
        #     self.rewards[agent_] = (
        #         -1
        #         * (player.score - self.game.MAX_SCORE + 1)
        #         * self.losing_penalty_multiplier
        #     )

        # self.terminations = {agent: True for agent in self.agents}

        # else:

        # Select next agent
        self.agent_selection = self.player_to_agent_dict[next_player]

        # Obtain new observations
        self.observations = self.get_observations(self.action_type)

        # Get rewards out of the state
        self.rewards = self.get_rewards()

        for player in self.game.players_that_lost:
            agent_ = self.player_to_agent_dict[player]
            self.rewards[agent_] += (
                -1
                * (player.score - self.game.MAX_SCORE + 1)
                * self.losing_penalty_multiplier
            )

        self.previous_scores_dict = self.get_current_scores()

        # Put the action mask in infos
        self.infos = self.get_infos(next_player, self.action_type)

        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

        self.logger.info(f"Next action: {self.action_type}")

        # return (
        #     self.observations,
        #     self.rewards,
        #     self.truncations,
        #     self.terminations,
        #     self.infos,
        # )

    def get_observations(self, action_type: ActionType):
        # NOTE can probably be made more efficient by saving the last space and adjust it

        # Initialize empty observation dict
        observation: dict = self.observation_space_base.empty_space()

        # Turn number
        observation["sub_round_number"] = np.array(
            self.game.sub_round, dtype=np.int32
        )
        observation["turn_number"] = np.array(self.game.turn, dtype=np.int32)
        observation["action_type"] = np.array(
            action_type_to_int(action_type), dtype=np.int32
        )

        # Scores
        scores = [player.score for player in self.game.players]
        observation["player_scores"] = np.array(scores, dtype=np.int32)

        # Laid down cards
        pile_space = []

        for player in self.game.players:
            adding = [self.card_to_number_dict[card] for card in player.pile]
            padding = [0] * (CARDS_PER_PLAYER - len(adding))
            adding.extend(padding)

            pile_space.extend(adding)

        observation["player_piles"] = np.array(pile_space, dtype=np.int32)

        observations_dict = {}
        for agent in self.agents:
            player_to_get_obs = self.agent_to_player_dict[agent]

            # Create a new dictionary for this agent
            agent_observation = copy.deepcopy(observation)

            # The cards in the players hand and the ones from players that have to play open
            hand_space = []
            for player in self.game.players:
                if player == player_to_get_obs or player.play_open:
                    adding = [
                        self.card_to_number_dict[card] for card in player.hand
                    ]
                else:
                    adding = []
                padding = [0] * (CARDS_PER_PLAYER - len(adding))
                adding.extend(padding)
                hand_space.extend(adding)

            agent_observation["player_hands"] = np.array(
                hand_space, dtype=np.int32
            )

            agent_observation = flatten(
                self.observation_space_base.observation_space_dict,
                agent_observation,
            )

            # Update the dictionary
            observations_dict[agent] = agent_observation

        return observations_dict

    def observe(self, agent):
        return self.observations[agent]

    def render(self):
        pass

    def close(self):
        pass

    def handle_action_for_player(
        self, player: Player, action_number
    ) -> tuple[Player, ActionType]:
        match action_number:
            case 0:
                new_player, action_type = player.toep()
            case 1:
                new_player, action_type = player.go_on()
            case 2:
                new_player, action_type = player.fold()
            case 3:
                new_player, action_type = player.call_vuile_was()
            case 4:
                new_player, action_type = player.dont_call_vuile_was()
            case 5:
                new_player, action_type = player.look_at_called_vuile_was()
            case 6:
                new_player, action_type = player.believe_vuile_was()
            case _:
                card = self.number_to_card_dict[action_number]
                new_player, action_type = player.play_card(card)

        return new_player, action_type

    def get_score_change(self) -> dict:
        current_scores = self.get_current_scores()

        score_changes_dict = {
            player: current_scores[player] - self.previous_scores_dict[player]
            for player in self.game.players
        }

        return score_changes_dict

    def get_current_scores(self) -> dict:
        return {player: player.score for player in self.game.players}

    def get_rewards(self) -> dict:
        score_change_dict = self.get_score_change()

        rewards_dict = {
            self.player_to_agent_dict[player]: -1 * change
            for player, change in score_change_dict.items()
        }

        return rewards_dict

    def get_mask(self, agent, action_type: ActionType, extra_mask=None):
        player = self.agent_to_player_dict[agent]
        mask = np.zeros(self.ACTION_SPACE_SIZE, dtype=np.int8)

        match action_type:
            case ActionType.GO_OR_FOLD:
                mask[1] = 1
                mask[2] = 1
            case ActionType.CALL_VUILE_WAS:
                mask[3] = 1
                mask[4] = 1
            case ActionType.CHECK_OR_TRUST:
                mask[5] = 1
                mask[6] = 1
            case ActionType.PLAY_CARD:
                legal_cards = player.legal_cards_to_play()
                legal_numbers = [
                    self.card_to_number_dict[card] for card in legal_cards
                ]

                if (
                    self.game.last_player_to_toep != player
                    and self.game.max_score < self.game.MAX_SCORE - 1
                ):
                    mask[0] = 1

                for number in legal_numbers:
                    mask[number] = 1

                # TODO: remove!!!!
                # mask[0] = 1

        if extra_mask is not None:
            mask[extra_mask] = 0

        return mask

    def get_infos(
        self,
        next_player: Player,
        action_type: ActionType,
        extra_mask: int = None,
    ):
        empty_mask = np.zeros(self.ACTION_SPACE_SIZE)
        next_agent = self.player_to_agent_dict[next_player]
        next_agent_mask = self.get_mask(next_agent, action_type, extra_mask)

        infos = {}

        for agent in self.agents:
            infos[agent] = {}

            if agent == next_agent:
                infos[agent]["action_mask"] = next_agent_mask
            else:
                infos[agent]["action_mask"] = empty_mask

        return infos

    def invalid_action(self, action: ActionType):
        mask = self.infos[self.agent_selection]["action_mask"]

        if mask[action] == 0:
            return True
        else:
            return False

    def invalid_action_penalty(self, action, action_type: ActionType):
        match action_type:
            case ActionType.GO_OR_FOLD:
                return abs(action - 1.5)
            case ActionType.CALL_VUILE_WAS:
                return abs(action - 3.5)
            case ActionType.CHECK_OR_TRUST:
                return abs(action - 5.5)
            case ActionType.PLAY_CARD:
                return 10  # NOTE: maybe not handy that toep is 0, but the cards start at 7

    @classmethod
    def env(cls, render_mode=None, **env_kwargs):
        """
        The env function often wraps the environment in wrappers by default.
        You can find full documentation for these methods
        elsewhere in the developer documentation.
        """

        internal_render_mode = (
            render_mode if render_mode != "ansi" else "human"
        )
        env = ToepEnv(**env_kwargs, render_mode=internal_render_mode)
        # This wrapper is only for environments which print results to the terminal
        if render_mode == "ansi":
            env = wrappers.CaptureStdoutWrapper(env)
        # this wrapper helps error handling for discrete action spaces
        # env = wrappers.AssertOutOfBoundsWrapper(env)
        # Provides a wide vareity of helpful user errors
        # Strongly recommended
        # env = wrappers.OrderEnforcingWrapper(env)

        # env = FlattenObservation(env)

        # env = aec_to_parallel(env)

        return env


def action_type_to_int(action_type: ActionType) -> int:
    match action_type:
        case ActionType.GO_OR_FOLD:
            return 0
        case ActionType.CALL_VUILE_WAS:
            return 1
        case ActionType.CHECK_OR_TRUST:
            return 2
        case ActionType.PLAY_CARD:
            return 3
