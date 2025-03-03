import pytest
from unittest.mock import patch

from catanatron.state_functions import player_has_rolled
from catanatron.models.actions import maritime_trade_possibilities
from catanatron.game import Game
from catanatron.state import (
    State,
    player_deck_replenish,
    player_num_resource_cards,
    yield_resources,
)
from catanatron.models.board import Board
from catanatron.models.enums import (
    ActionPrompt,
    Resource,
    BuildingType,
    ActionType,
    Action,
    WHEAT,
    YEAR_OF_PLENTY,
)
from catanatron.models.player import Color, SimplePlayer
from catanatron.models.decks import ResourceDeck


def test_initial_build_phase():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    actions = []
    while not any(
        a.action_type == ActionType.ROLL for a in game.state.playable_actions
    ):
        actions.append(game.play_tick())

    p0_color = game.state.colors[0]
    assert (
        actions[0].action_type == ActionType.BUILD_SETTLEMENT
        and actions[0].color == p0_color
    )
    assert (
        actions[1].action_type == ActionType.BUILD_ROAD and actions[1].color == p0_color
    )
    assert (
        actions[2].action_type == ActionType.BUILD_SETTLEMENT
        and actions[2].color != p0_color
    )
    assert (
        actions[3].action_type == ActionType.BUILD_ROAD and actions[3].color != p0_color
    )
    assert (
        actions[4].action_type == ActionType.BUILD_SETTLEMENT
        and actions[4].color != p0_color
    )
    assert (
        actions[5].action_type == ActionType.BUILD_ROAD and actions[5].color != p0_color
    )
    assert (
        actions[6].action_type == ActionType.BUILD_SETTLEMENT
        and actions[6].color == p0_color
    )
    assert (
        actions[7].action_type == ActionType.BUILD_ROAD and actions[7].color == p0_color
    )
    assert (
        game.state.current_prompt == ActionPrompt.PLAY_TURN
        and game.state.current_player().color == p0_color
    )

    assert game.state.player_state["P0_ACTUAL_VICTORY_POINTS"] == 2
    assert game.state.player_state["P1_ACTUAL_VICTORY_POINTS"] == 2
    assert game.state.player_state["P0_VICTORY_POINTS"] == 2
    assert game.state.player_state["P1_VICTORY_POINTS"] == 2

    # assert there are 4 houses and 4 roads
    settlements = [
        building
        for building in game.state.board.buildings.values()
        if building[1] == BuildingType.SETTLEMENT
    ]
    assert len(settlements) == 4

    # assert should be house-road pairs, or together
    paths = game.state.board.continuous_roads_by_player(players[0].color)
    assert len(paths) == 1 or (
        len(paths) == 2 and len(paths[0]) == 1 and len(paths[1]) == 1
    )

    # assert should have resources from last house.
    # can only assert <= 3 b.c. player might place on a corner desert
    assert player_num_resource_cards(game.state, players[0].color) <= 3
    assert player_num_resource_cards(game.state, players[1].color) <= 3


def test_can_play_for_a_bit():  # assert no exception thrown
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    for _ in range(10):
        game.play_tick()


@patch("catanatron.state.roll_dice")
def test_seven_cards_dont_trigger_discarding(fake_roll_dice):
    fake_roll_dice.return_value = (1, 6)
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]

    # Play initial build phase
    game = Game(players)
    while not any(
        a.action_type == ActionType.ROLL for a in game.state.playable_actions
    ):
        game.play_tick()

    until_seven = 7 - player_num_resource_cards(game.state, players[1].color)
    player_deck_replenish(game.state, players[1].color, WHEAT, until_seven)
    assert player_num_resource_cards(game.state, players[1].color) == 7
    game.play_tick()  # should be player 0 rolling.

    print(game.state.playable_actions)
    assert not any(
        a.action_type == ActionType.DISCARD for a in game.state.playable_actions
    )


@patch("catanatron.state.roll_dice")
def test_rolling_a_seven_triggers_discard_mechanism(fake_roll_dice):
    fake_roll_dice.return_value = (1, 6)
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    while not any(
        a.action_type == ActionType.ROLL for a in game.state.playable_actions
    ):
        game.play_tick()

    until_nine = 9 - player_num_resource_cards(game.state, players[1].color)
    player_deck_replenish(game.state, players[1].color, WHEAT, until_nine)
    assert player_num_resource_cards(game.state, players[1].color) == 9
    game.play_tick()  # should be player 0 rolling.

    assert len(game.state.playable_actions) == 1
    assert game.state.playable_actions == [
        Action(players[1].color, ActionType.DISCARD, None)
    ]

    game.play_tick()
    assert player_num_resource_cards(game.state, players[1].color) == 5


@patch("catanatron.state.roll_dice")
def test_end_turn_goes_to_next_player(fake_roll_dice):
    fake_roll_dice.return_value = (1, 2)  # not a 7

    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)
    actions = []
    while not any(
        a.action_type == ActionType.ROLL for a in game.state.playable_actions
    ):
        actions.append(game.play_tick())

    p0_color = game.state.colors[0]
    p1_color = game.state.colors[1]
    assert (
        game.state.current_prompt == ActionPrompt.PLAY_TURN
        and game.state.current_player().color == p0_color
    )
    assert game.state.playable_actions == [Action(p0_color, ActionType.ROLL, None)]

    game.execute(Action(p0_color, ActionType.ROLL, None))
    assert game.state.current_prompt == ActionPrompt.PLAY_TURN
    assert game.state.current_player().color == p0_color
    assert player_has_rolled(game.state, p0_color)
    assert Action(p0_color, ActionType.END_TURN, None) in game.state.playable_actions

    game.execute(Action(p0_color, ActionType.END_TURN, None))
    assert game.state.current_prompt == ActionPrompt.PLAY_TURN
    assert game.state.current_player().color == p1_color
    assert not player_has_rolled(game.state, p0_color)
    assert not player_has_rolled(game.state, p1_color)
    assert game.state.playable_actions == [Action(p1_color, ActionType.ROLL, None)]


# ===== Development Cards


def test_play_year_of_plenty_not_enough_resources():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    player_to_act = players[0]
    game = Game(players)
    game.state.resource_deck = ResourceDeck()
    player_deck_replenish(game.state, player_to_act.color, YEAR_OF_PLENTY)

    action_to_execute = Action(
        player_to_act.color,
        ActionType.PLAY_YEAR_OF_PLENTY,
        [Resource.ORE, Resource.WHEAT],
    )

    with pytest.raises(ValueError):  # not enough cards in bank
        game.execute(action_to_execute)


def test_play_year_of_plenty_no_year_of_plenty_card():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)

    action_to_execute = Action(
        players[0].color, ActionType.PLAY_YEAR_OF_PLENTY, [Resource.ORE, Resource.WHEAT]
    )

    with pytest.raises(ValueError):  # no year of plenty card
        game.execute(action_to_execute)


def test_play_monopoly_no_monopoly_card():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    game = Game(players)

    action_to_execute = Action(players[0].color, ActionType.PLAY_MONOPOLY, Resource.ORE)

    with pytest.raises(ValueError):  # no monopoly
        game.execute(action_to_execute)


# ===== Yield Resources
def test_yield_resources():
    board = Board()
    resource_deck = ResourceDeck.starting_bank()

    tile = board.map.tiles[(0, 0, 0)]
    if tile.resource is None:  # is desert
        tile = board.map.tiles[(-1, 0, 1)]

    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    payout, depleted = yield_resources(board, resource_deck, tile.number)
    assert len(depleted) == 0
    assert payout[Color.RED].count(tile.resource) >= 1


def test_yield_resources_two_settlements():
    board = Board()
    resource_deck = ResourceDeck.starting_bank()

    tile, edge2, node2 = board.map.tiles[(0, 0, 0)], (4, 5), 5
    if tile.resource is None:  # is desert
        tile, edge2, node2 = board.map.tiles[(-1, 0, 1)], (4, 15), 15

    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 4))
    board.build_road(Color.RED, edge2)
    board.build_settlement(Color.RED, node2)
    payout, depleted = yield_resources(board, resource_deck, tile.number)
    assert len(depleted) == 0
    assert payout[Color.RED].count(tile.resource) >= 2


def test_yield_resources_two_players_and_city():
    board = Board()
    resource_deck = ResourceDeck.starting_bank()

    tile, edge1, edge2, red_node, blue_node = (
        board.map.tiles[(0, 0, 0)],
        (2, 3),
        (3, 4),
        4,
        0,
    )
    if tile.resource is None:  # is desert
        tile, edge1, edge2, red_node, blue_node = (
            board.map.tiles[(1, -1, 0)],
            (9, 2),
            (9, 8),
            8,
            6,
        )

    # red has one settlements and one city on tile
    board.build_settlement(Color.RED, 2, initial_build_phase=True)
    board.build_road(Color.RED, edge1)
    board.build_road(Color.RED, edge2)
    board.build_settlement(Color.RED, red_node)
    board.build_city(Color.RED, red_node)

    # blue has a city in tile
    board.build_settlement(Color.BLUE, blue_node, initial_build_phase=True)
    board.build_city(Color.BLUE, blue_node)
    payout, depleted = yield_resources(board, resource_deck, tile.number)
    assert len(depleted) == 0
    assert payout[Color.RED].count(tile.resource) >= 3
    assert payout[Color.BLUE].count(tile.resource) >= 2


def test_empty_payout_if_not_enough_resources():
    board = Board()
    resource_deck = ResourceDeck.starting_bank()

    tile = board.map.tiles[(0, 0, 0)]
    if tile.resource is None:  # is desert
        tile = board.map.tiles[(-1, 0, 1)]

    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_city(Color.RED, 3)
    resource_deck.draw(18, tile.resource)

    payout, depleted = yield_resources(board, resource_deck, tile.number)
    assert depleted == [tile.resource]
    assert Color.RED not in payout or payout[Color.RED].count(tile.resource) == 0


def test_can_trade_with_port():
    players = [SimplePlayer(Color.RED)]

    state = State(players)
    state.board.build_settlement(Color.RED, 26, initial_build_phase=True)

    port_tile = state.board.map.tiles[(3, -3, 0)]  # port with node_id=25,26
    resource_out = port_tile.resource or Resource.WHEAT
    num_out = 3 if port_tile.resource is None else 2
    player_deck_replenish(state, Color.RED, resource_out.value, num_out)

    possibilities = maritime_trade_possibilities(state, Color.RED)
    assert len(possibilities) == 4


def test_second_placement_takes_cards_from_bank():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.WHITE),
        SimplePlayer(Color.ORANGE),
    ]
    game = Game(players)
    assert len(game.state.resource_deck.to_array()) == 19 * 5

    while not any(
        a.action_type == ActionType.ROLL for a in game.state.playable_actions
    ):
        game.play_tick()

    assert len(game.state.resource_deck.to_array()) < 19 * 5
