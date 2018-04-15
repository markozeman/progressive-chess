import heapq
import random
import chess
from operator import itemgetter
# import time
# import csv


'''
def read_examples():
    with open('progressive_checkmates.csv', 'r') as f:
        reader = csv.reader(f)
        return list(map(lambda x: x[1], list(reader)[1:]))


def image_url(fen):
    fen = fen.split(' ')[0]
    base_url = 'https://backscattering.de/web-boardimage/board.png?fen='
    url = base_url + fen
    return url
'''


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


def back2fen(pos_2):
    new_pos = []
    for row in pos_2:
        s = ''
        count = 0
        j = 0
        for i in range(len(row)):
            if i >= j:
                if row[i] == '.':
                    while i + count < len(row) and row[i + count] == '.':
                        count += 1
                    s += str(count)
                    j = i + count
                else:
                    count = 0
                    s += row[i]
        new_pos.append(s)
    return new_pos


def f(g, hevristics):
    return g - sum(hevristics)


class Seminar1:
    def __init__(self):
        self.visited = {}
        self.char2num = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        self.opposite_king = None
        self.new_position = None
        self.king_directions = None
        self.mat_square = None
        self.curr_figure = None
        self.chessboard = None

    def updated_position(self, pos, move, turn):
        pos_ = pos.split('/')
        pos = ''.join(map(lambda c: int(c) * '.' if c.isdigit() else c, pos))
        pos_split = pos.split('/')

        move = str(move)
        old_row = abs(int(move[1]) - 8)
        old_col = self.char2num[move[0]]
        new_row = abs(int(move[3]) - 8)
        new_col = self.char2num[move[2]]

        if len(move) == 5:  # promotion
            if turn == 'b':
                pos_split[new_row] = pos_split[new_row][:new_col] + move[4] + pos_split[new_row][new_col + 1:]
            else:
                pos_split[new_row] = pos_split[new_row][:new_col] + move[4].upper() + pos_split[new_row][new_col + 1:]
        else:
            pos_split[new_row] = pos_split[new_row][:new_col] + pos_split[old_row][old_col] + pos_split[new_row][new_col+1:]

        pos_split[old_row] = pos_split[old_row][:old_col] + '.' + pos_split[old_row][old_col+1:]

        pos_2_rows = itemgetter(old_row, new_row)(pos_split)

        pos_[old_row], pos_[new_row] = back2fen(pos_2_rows)

        return '/'.join(pos_)

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

    def position_on_board(self, square_str):
        a, n = square_str
        num = (int(n) - 1) * 8
        num += self.char2num[a]
        return num

    def h_mat_square(self, board, factor=1.0):
        squares_attackers = list(map(lambda s: len(list(board.attackers(not self.opposite_king[3], s))), self.mat_square))
        return factor * sum(squares_attackers)

    def h_different_figures(self, board, factor=1.0):
        self.curr_figure = str(board.piece_at(self.new_position)).lower()
        d = {'p': 3, 'n': 4, 'b': 5, 'r': 2, 'q': 6, 'k': 1}
        return factor * d[self.curr_figure]

    def h_distance2king(self, factor=1.0):
        return -factor * chess.square_distance(self.opposite_king[2], self.new_position)

    def h_opposite_king_directions(self, factor=1.0):
        is_rqb = self.curr_figure == 'r' or self.curr_figure == 'q' or self.curr_figure == 'b'
        if self.new_position in self.king_directions and is_rqb:
            return factor * 1
        return 0

    def save_new_position(self, move):
        new_pos = str(move)[2:4]
        self.new_position = self.position_on_board(new_pos)

    def a_star(self, board, curr_moves, fen):
        position, turn, _, _, _, num_moves = fen.split(' ')
        num_moves = int(num_moves)

        queue = []
        heapq.heappush(queue, (0, {'pos': position, 'g': curr_moves, 'path': ''}))

        while queue:
            node = heapq.heappop(queue)[1]
            curr_moves = node['g']
            self.visited[node['pos']] = curr_moves
            board.set_fen(position2fen(node['pos'], turn))

            null_move(board)  # to check for checkmate
            if board.is_checkmate():
                return node
            board.pop()  # reverse null move

            # check if new g exceeds number of available moves
            g = curr_moves + 1
            if g > num_moves:
                continue

            for move in list(board.legal_moves):
                board.push(move)

                # if it is check and not the last move, do not add to queue
                if board.is_check() and g != num_moves:
                    board.pop()
                    continue

                # check if it was not already been visited
                pos = self.updated_position(node['pos'], move, turn)
                if pos in self.visited and g >= self.visited[pos]:
                    board.pop()
                    continue

                self.save_new_position(move)

                hevs = [self.h_mat_square(board), self.h_different_figures(board, factor=0.1),
                        self.h_distance2king(factor=0.1), self.h_opposite_king_directions(factor=2)]

                f_est = f(g, hevs) + random.uniform(0, 0.001)

                new_path = node['path'] + str(move) + ';'
                heapq.heappush(queue, (f_est, {'pos': pos, 'g': g, 'path': new_path}))

                board.pop()

    def solve(self, fen):
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

        final_node = self.a_star(board, curr_moves, fen)
        if final_node is not None:
            return path_in_right_form(final_node['path'])

    def studentId(self):
        return '63140295'


'''
if __name__ == '__main__':
    examples = read_examples()

    n = 60
    solved = 0

    sah = Seminar1()
    start_all = time.time()
    for i, example in enumerate(examples):
        if i >= n:
            break

        # print(example[-3:], image_url(example))
        start = time.time()
        solution = sah.solve(example)
        end = time.time()
        print(solution)
        if solution is not None:
            solved += 1
        print(i, '-- Elapsed time: ', round(end - start, 1), 's\n')
    end_all = time.time()

    print('Whole elapsed time: ', round(end_all - start_all, 1), 's\n')
    print('Average time: ', round((end_all - start_all) / n, 1), 's\n')
    print('Solved: ', solved, '/', n)
'''
