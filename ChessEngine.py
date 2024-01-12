from stockfish import Stockfish
from ConnectChessDotCom import get_fen


def orchestrate():
    stockfish = Stockfish(path="stockfish/stockfish-windows-x86-64.exe")
    stockfish.set_elo_rating(1350)
    stockfish.set_depth(15)
    stockfish.set_fen_position(get_fen('chantelletay'))

    print(f'Best move is: {stockfish.get_best_move()}')
    print(f'Top 5 moves are: {stockfish.get_top_moves(5)}')

