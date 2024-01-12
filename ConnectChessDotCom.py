from chessdotcom import get_player_profile, Client
import chessdotcom


def orchestrate():
    pass


def get_fen(player='chantelletay'):
    Client.request_config["headers"]["User-Agent"] = (
        "My Python Application. "
        "Contact me at maxbruce9@gmail.com"
    )
    response = get_player_profile("Bruce_Max_Dickie")
    print(response.json)

    chess_response = chessdotcom.get_player_current_games('Bruce_Max_Dickie')

    for game in chess_response.games:
        if player in game.pgn:
            print(game)
            return game.fen
