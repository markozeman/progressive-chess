import heapq
import random
import chess
import urllib.request as url_req
import csv
import time


def test_html():
    board = chess.Board()

    fen1 = 'rnbBkbnr/pp1p1ppp/8/8/3p4/8/PPP1PPPP/RN1QKBNR b 4'
    corr_fen = correct_fen(fen1)
    print(corr_fen)

    board.set_fen(corr_fen)

    print(whose_turn(board))
    board.push(chess.Move.null())  # null move
    print(whose_turn(board))

    print(board.is_check())
    print(board.is_checkmate())
    print(board.is_game_over())

    print(board)
    save_image(board, 'test.png')

    print(board.is_attacked_by(chess.BLACK, chess.D8))

    attackers = board.attackers(chess.WHITE, chess.D3)
    print(attackers)
    print(list(attackers))
    print(chess.D1 in attackers)

    print(board.attacks(chess.D8))

    print(board.piece_at(chess.D8))

    print(board.pieces(chess.BISHOP, chess.WHITE))


def read_examples():
    with open('progressive_checkmates.csv', 'r') as f:
        reader = csv.reader(f)
        return list(map(lambda x: x[1], list(reader)[1:]))


def save_image(fen, png_filename='example'):
    fen = fen.split(' ')[0]
    base_url = 'https://backscattering.de/web-boardimage/board.png?fen='
    url = base_url + fen
    return url
    # png_image = url_req.urlopen(url).read()
    # with open('boards/' + png_filename + '.png', 'wb') as f:
    #     f.write(png_image)


def path_in_right_form(path):
    return ';'.join(list(map(lambda x: x[:2] + '-' + x[2:], path[:-1].split(';'))))


def correct_fen(fen):
    fen_split = fen.split(' ')
    fen_split.insert(2, '- - 0')
    return ' '.join(fen_split)


def position2fen(pos, turn):
    return ' '.join([pos, turn, '- - 0 99'])


def whose_turn(board):
    return 'white' if board.turn else 'black'


def null_move(board):
    board.push(chess.Move.null())


def find_opposite_king(board, turn):
    attacked_wb = chess.BLACK if turn == 'w' else chess.WHITE

    # if not (list(board.pieces(chess.KING, attacked_wb))):
    #     return -99

    opposite_king = list(board.pieces(chess.KING, attacked_wb))[0]
    row_index = chess.square_rank(opposite_king)
    column_index = chess.square_file(opposite_king)

    return [row_index, column_index, opposite_king, attacked_wb]


def find_checkmate_square(row, column):
    positions = []
    positions.append([row + 1, column - 1])
    positions.append([row, column - 1])
    positions.append([row - 1, column - 1])
    positions.append([row + 1, column])
    positions.append([row, column])
    positions.append([row - 1, column])
    positions.append([row + 1, column + 1])
    positions.append([row, column + 1])
    positions.append([row - 1, column + 1])
    positions = list(filter(lambda pos: 0 <= pos[0] < 8 and 0 <= pos[1] < 8, positions))
    squares = list(map(lambda pos: chess.square(pos[1], pos[0]), positions))
    return squares


def position_on_board(square_str):
    d = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
    a, n = square_str
    num = (int(n)-1) * 8
    num += d[a]
    return num


def f(g, hevristics):
    return g - sum(hevristics)


class Seminar1:
    def __init__(self):
        self.visited = {}
        self.opposite_king = None
        self.new_position = None
        self.king_directions = None
        self.mat_square = None
        self.curr_figure = None

    def save_king_directions(self):
        king_num = self.opposite_king[2]
        empty_down = king_num // 8
        empty_up = 7 - empty_down
        empty_left = king_num % 8
        empty_right = 7 - empty_left
        fen_only_queen = '8/' * empty_up + str(empty_left) + 'q' + str(empty_right) + '/8' * empty_down + ' b 1'
        b = chess.Board()
        b.set_fen(correct_fen(fen_only_queen))
        self.king_directions = list(b.attacks(king_num))

    def h_mat_square(self, board, factor=1.0):
        squares_attackers = list(map(lambda s: len(list(board.attackers(not self.opposite_king[3], s))), self.mat_square))
        return factor * sum(squares_attackers)

    def h_different_figures(self, board, factor=1.0):
        self.curr_figure = str(board.piece_at(self.new_position)).lower()
        d = {'p': 2, 'n': 4, 'b': 5, 'r': 2, 'q': 7, 'k': 1}
        return factor * d[self.curr_figure]

    def h_distance2king(self, factor=1.0):
        return -factor * chess.square_distance(self.opposite_king[2], self.new_position)

    def h_opposite_king_directions(self, factor=1.0):
        is_b_or_q = self.curr_figure == 'r' or self.curr_figure == 'q'
        if self.new_position in self.king_directions and is_b_or_q:
            return factor * 1
        return 0

    def h_square_color(self, board, move):
        pass

    def save_new_position(self, move):
        move = str(move)
        new_pos = move[-2:] if len(move) == 4 else move[2:4]
        self.new_position = position_on_board(new_pos)

    def a_star(self, board, curr_moves, fen, max_time):
        start = time.time()
        position, turn, _, _, _, num_moves = fen.split(' ')
        num_moves = int(num_moves)

        queue = []
        heapq.heappush(queue, (0, {'pos': position, 'f': 0, 'g': curr_moves, 'path': ''}))

        while queue and time.time() - start < max_time:
            node = heapq.heappop(queue)[1]
            curr_moves = node['g']
            board.set_fen(position2fen(node['pos'], turn))
            self.visited[node['pos']] = 0       ###  curr_moves !!!!!

            null_move(board)  # to check for check and checkmate
            if board.is_check() and curr_moves < num_moves:     # se sme zakomentirat?
                continue

            if board.is_checkmate():
                return node

            board.pop()  # reverse null move

            for move in list(board.legal_moves):
                board.push(move)

                # check if new g exceeds number of available moves
                g = curr_moves + 1      # out of the loop!!!
                if g > int(num_moves):
                    board.pop()
                    continue

                # if it is check and not the last move, do not add to queue
                if board.is_check() and curr_moves != num_moves - 1:
                    board.pop()
                    continue

                # check if it was not already been visited
                pos = board.board_fen()
                if pos in self.visited and curr_moves >= self.visited[pos]:  # second condition always true, if 0 put into self.visited
                    board.pop()
                    continue


                self.save_new_position(move)

                hevs = [self.h_mat_square(board), self.h_different_figures(board, factor=0.1),
                        self.h_distance2king(factor=0.5), self.h_opposite_king_directions(factor=2)]

                f_est = f(g, hevs) + random.uniform(0, 0.001)

                new_path = node['path'] + str(move) + ';'
                heapq.heappush(queue, (f_est, {'pos': pos, 'f': f_est, 'g': g, 'path': new_path}))

                board.pop()


    def solve(self, fen, max_time=20):
        self.__init__()

        fen = correct_fen(fen)
        position, turn, _, _, _, num_moves = fen.split(' ')

        board = chess.Board()
        board.set_fen(fen)

        self.opposite_king = find_opposite_king(board, turn)
        self.mat_square = find_checkmate_square(self.opposite_king[0], self.opposite_king[1])
        self.save_king_directions()

        curr_moves = 0
        self.visited[position] = curr_moves
        if whose_turn(board)[0] != turn:
            null_move(board)

        final_node = self.a_star(board, curr_moves, fen, max_time)
        if final_node is not None:
            return path_in_right_form(final_node['path'])

    def studentId(self):
        return '63140295'



if __name__ == '__main__':
    # test_html()

    examples = read_examples()

    n = 10

    sah = Seminar1()
    start_all = time.time()
    for i, example in enumerate(examples):
        if i >= n:
            break

        print(example[-3:], save_image(example))
        start = time.time()
        solution = sah.solve(example)
        end = time.time()
        print(solution)
        print(i, '-- Elapsed time: ', round(end - start, 1), 's\n')
    end_all = time.time()

    print('Whole elapsed time: ', round(end_all - start_all, 1), 's\n')
    print('Average time: ', round((end_all - start_all) / n, 1), 's\n')


