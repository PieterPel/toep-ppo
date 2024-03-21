class NotEnoughPlayersError(Exception):
    """Gets thrown if a game doesnt have enough players"""


class TooManyPlayersError(Exception):
    """Gets thrown if a game has too much players"""
