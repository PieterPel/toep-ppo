from enum import Enum, auto
import itertools
import random
import math
import copy
import logging

from toeppo.errors import NotEnoughPlayersError, TooManyPlayersError

# Action space: Toep, Fold, Mee, Lay card (x32)
CARDS_PER_PLAYER = 4


class Suit(Enum):
    HEARTS = auto()
    DIAMONDS = auto()
    SPADES = auto()
    CLUBS = auto()


class Rank(Enum):
    JACK = auto()
    QUEEN = auto()
    KING = auto()
    ACE = auto()
    SEVEN = auto()
    EIGHT = auto()
    NINE = auto()
    TEN = auto()


class ActionType(Enum):
    PLAY_CARD = auto()
    GO_OR_FOLD = auto()
    CHECK_OR_TRUST = auto()
    CALL_VUILE_WAS = auto()
    LOST = auto()


class Card:
    rank_to_value = {
        Rank.JACK: 3,
        Rank.QUEEN: 4,
        Rank.KING: 5,
        Rank.ACE: 6,
        Rank.SEVEN: 7,
        Rank.EIGHT: 8,
        Rank.NINE: 9,
        Rank.TEN: 10,
    }

    def __init__(self, suit: Suit, rank: Rank):
        self.suit = suit
        self.rank = rank

    @property
    def value(self):
        return self.rank_to_value[self.rank]

    def __eq__(self, other):
        return self.suit == other.suit and self.rank == other.rank


class CardCollection:

    def __init__(self):
        self.cards = []

    def add_card(self, card: Card):
        self.cards.append(card)

    def clear(self):
        self.cards = []

    def shuffle(self):
        random.shuffle(self.cards)

    def __getitem__(self, index):
        return self.cards[index]

    def __len__(self):
        return len(self.cards)


class Deck(CardCollection):
    def __init__(self):
        ranks = Rank.__members__.values()
        suits = Suit.__members__.values()

        self.cards = [
            Card(suit, rank) for suit, rank in itertools.product(suits, ranks)
        ]

    def draw_card(self):
        return self.cards.pop()

    @classmethod
    def shuffled_deck(cls):
        deck = cls()
        deck.shuffle()
        return deck


class PlayerPile(CardCollection):
    """"""


class PlayerHand(CardCollection):
    """"""

    @property
    def vuile_was(self):
        if len(self) < CARDS_PER_PLAYER:
            return False
        else:
            seen_seven = False
            for card in self:
                if card.value > 7:
                    return False
                if seen_seven and card.value >= 7:
                    return False
                if card.rank == Rank.SEVEN:
                    seen_seven = True

            return True


class Player:
    def __init__(self, name):
        self.name = name
        self.pile = PlayerPile()
        self.hand = PlayerHand()
        self.score = 0
        self.pussy_points = 0
        self.active = False
        self.game = None
        self.play_open = False  # TODO: further implement this logic in the observation space maybe

    def enter_game(self, game: "ToepGame"):
        self.game = game

    def reset_cards(self):
        self.pile = PlayerPile()
        self.hand = PlayerHand()

    def legal_cards_to_play(self):
        cards_list = [card for card in self.hand]

        if self.game.leading_suit is None:
            legal_cards = cards_list
        else:
            legal_cards = [
                card
                for card in cards_list
                if card.suit == self.game.leading_suit
            ]

        if legal_cards == []:
            legal_cards = cards_list

        return legal_cards

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return (
            isinstance(other, Player) and self.name == other.name
        )  # TODO: should probably be able to have instances with the same name not be equal

    def __str__(self):
        return self.name

    # Actions
    def play_card(self, card: Card):
        self.pile.add_card(card)

        return self.game.handle_played_card(self, card)

    def toep(self):
        return self.game.handle_toep(self)

    def fold(self):
        return self.game.handle_fold(self)

    def go_on(self):
        return self.game.handle_go()

    def look_at_called_vuile_was(self):
        return self.game.handle_looked_vuile_was(self)

    def belive_vuile_was(self):
        return self.game.handle_believed_vuile_was(self)

    def call_vuile_was(self):
        return self.game.handle_called_vuile_was(self)

    def dont_call_vuile_was(self):
        return self.game.handle_not_called_vuile_was(self)


class ToepGame:
    MAX_SCORE = 15

    def __init__(self, n_players: int):
        self.logger = logging.getLogger(__name__)

        self.deck = Deck()

        if n_players < 2:
            raise NotEnoughPlayersError()
        elif n_players > math.floor(len(self.deck) / CARDS_PER_PLAYER):
            raise TooManyPlayersError()

        self.players = [
            Player(f"player_{str(i)}") for i in range(1, n_players + 1)
        ]
        self.set_players_game()

        self.dealing_player = self.players[0]
        self.active_player = None
        self.turn = 0
        self.sub_round = 0
        self.leading_suit = None
        self.stake = 0
        self.called_vuile_was = None
        self.players_that_looked = []
        self.last_player_to_toep = None

    def set_players_game(self):
        for player in self.players:
            player.enter_game(self)

    def start_round(self) -> tuple[Player, ActionType]:
        self.logger.info("Starting a round")

        if self.ended_game:
            return (
                self.losing_players,
                ActionType.LOST,
            )  # TODO: this returns a list now, not the best code since all other functions return a single player. Maybe just check in the environment for a lost game?

        self.deck.shuffle()
        self.distribute_cards()
        self.sub_round = 0
        self.stake = 1
        self.alive_players = copy.copy(self.players)
        self.update_players_dict()
        self.active_player = self.next_player_dict[self.dealing_player]
        self.last_player_to_toep = None

        if not self.armoe:
            return self.start_vuile_was_round()
        else:
            return self.start_sub_round()

    def start_vuile_was_round(self) -> tuple[Player, ActionType]:
        self.logger.info("Starting a vuile was round")

        return self.active_player, ActionType.CALL_VUILE_WAS

    def start_sub_round(self) -> tuple[Player, ActionType]:
        self.logger.info("Starting a sub round")

        self.last_player_of_sub_round = self.last_player_dict[
            self.active_player
        ]
        self.sub_round += 1
        self.turn = 1
        self.leading_suit = None

        return self.active_player, ActionType.PLAY_CARD

    def determine_sub_round_winner(self) -> tuple[Player, Card]:
        # Compare the last played cards of alive players
        best_player: Player = self.alive_players[0]
        best_card: Card = best_player.pile[-1]

        for player in self.alive_players[1:]:
            last_card: Card = player.pile[-1]

            if (
                last_card.suit == self.leading_suit
                and last_card.value > best_card.value
            ):
                best_player = player
                best_card = last_card

        return best_player, best_card

    def end_sub_round(self) -> tuple[Player, ActionType]:
        self.logger.info("Ending a sub round")

        self.winning_player, self.winning_card = (
            self.determine_sub_round_winner()
        )
        self.dealing_player = self.winning_player

        if self.sub_round == CARDS_PER_PLAYER:
            return self.end_round()
        else:
            return self.start_sub_round()

    def end_round(self) -> tuple[Player, ActionType]:
        self.logger.info("Ending a round")

        if self.winning_card.rank == Rank.JACK:
            self.stake *= 2

        self.give_scores_at_end_of_round()

        self.reset_players()

        return self.start_round()

    def give_scores_at_end_of_round(self):
        for player in self.alive_players:
            if player != self.winning_player:
                player.score += self.stake

    def reset_players(self):
        for player in self.players:
            player.reset_cards()
            player.play_open = False

    def distribute_cards(self):
        for _ in range(CARDS_PER_PLAYER):
            for player in self.players:
                drawn_card = self.deck.draw_card()
                player.hand.add_card(drawn_card)

    def handle_looked_vuile_was(
        self, player: Player
    ) -> tuple[Player, ActionType]:
        self.logger.info(f"{player} looked at the vuile was")

        self.players_that_looked.append(player)

        if player == self.last_player_to_look_or_believe:
            self.handle_vuile_was_end(player)
        else:
            return self.next_player_dict[player], ActionType.CHECK_OR_TRUST

    def handle_called_vuile_was(
        self, player: Player
    ) -> tuple[Player, ActionType]:
        self.logger.info(f"{player} called a vuile was")

        self.called_vuile_was = player
        self.last_player_to_look_or_believe = self.last_player_dict[player]

        return self.next_player_dict[player], ActionType.CHECK_OR_TRUST

    def handle_not_called_vuile_was(
        self, player: Player
    ) -> tuple[Player, ActionType]:
        self.logger.info(f"{player} did not call a vuile was")

        if player == self.last_player_dict[self.active_player]:
            self.start_sub_round()
        else:
            return self.next_player_dict[player], ActionType.CALL_VUILE_WAS

    def handle_believed_vuile_was(
        self, player: Player
    ) -> tuple[Player, ActionType]:
        self.logger.info(f"{player} believed the vuile was")

        if player == self.last_player_to_look_or_believe:
            self.handle_vuile_was_end(player)

        return self.next_player_dict[player], ActionType.CHECK_OR_TRUST

    def handle_vuile_was_end(
        self, last_player: Player
    ) -> tuple[Player, ActionType]:
        if self.called_vuile_was.hand.vuile_was:
            for player in self.players_that_looked:
                player.score += 1
        else:
            self.called_vuile_was.score += 1
            self.called_vuile_was.play_open = True

        if last_player == self.last_player_dict[self.active_player]:
            self.start_sub_round()
        else:
            return self.next_player_dict[player], ActionType.CALL_VUILE_WAS

    def handle_played_card(
        self, player: Player, card: Card
    ) -> tuple[Player, ActionType]:
        self.logger.info(f"{player} played a {card}")

        self.active_player = self.next_player_dict[self]

        if self.turn == 1:
            self.leading_suit = card.suit

        self.turn += 1

        if player == self.last_player_of_sub_round:
            self.end_sub_round()
        else:
            self.active_player = self.next_player_dict[player]

        return self.next_player_dict[player], ActionType.PLAY_CARD

    def handle_fold(self, player: Player) -> tuple[Player, ActionType]:
        self.logger.info(f"{player} folded")

        player.score += self.stake

        if self.last_player_to_toep is None:
            player.pussy_points += 1

        self.alive_players.remove(player)

        if player == self.last_player_of_sub_round:
            self.last_player_of_sub_round = self.last_player_dict[player]

        if player == self.last_player_dict[self.active_player]:
            self.update_players_dict()
            return self.handle_ended_go_or_fold_round
        else:
            self.update_players_dict()
            return self.next_player_dict[player], ActionType.GO_OR_FOLD

    def handle_go(self, player: Player) -> tuple[Player, ActionType]:
        self.logger.info(f"{player} goes with the toep")

        if player == self.last_player_dict[self.active_player]:
            return self.handle_ended_go_or_fold_round()

        return self.next_player_dict[player], ActionType.GO_OR_FOLD

    def handle_toep(self, player: Player) -> tuple[Player, ActionType]:
        self.logger.info(f"{player} toeps")

        return self.next_player_dict[player], ActionType.GO_OR_FOLD

    def handle_ended_go_or_fold_round(self) -> tuple[Player, ActionType]:
        if len(self.alive_players) == 1:
            self.end_round()

        self.stake += 1
        self.last_player_to_toep = self.active_player

        return self.active_player, ActionType.PLAY_CARD

    def update_players_dict(self):
        self.next_player_dict = {
            player: self.alive_players[index + 1]
            for index, player in enumerate(self.alive_players[:-1])
        }
        self.next_player_dict[self.alive_players[-1]] = self.alive_players[0]

        self.last_player_dict = {
            player: self.alive_players[index - 1]
            for index, player in enumerate(self.alive_players[1:])
        }
        self.last_player_dict[self.alive_players[0]] = self.alive_players[-1]

    @property
    def max_score(self) -> int:
        highest_score = self.players[0].score

        for player in self.players[1:]:
            if player.score > highest_score:
                highest_score = player.score

        return highest_score

    @property
    def losing_players(self) -> list[Player]:
        players_above_max_score = []

        for player in self.players:
            if player.score >= self.MAX_SCORE:
                players_above_max_score.append(player)

        return player

    @property
    def ended_game(self) -> bool:
        if self.max_score >= self.MAX_SCORE:
            return True
        else:
            return False

    @property
    def armoe(self) -> bool:
        if self.max_score == self.MAX_SCORE - 1:
            return True
        else:
            return False
