import socketserver
import tensorflow as tf
import math
import numpy as np
import os.path
import logging


meta_data_cols = 4
num_of_rows = 8
num_of_cols = 4


class NeuralDraught:
    single_game_history = []
    training_history = {}
    game_number = 0
    max_game_number = 1000
    training_number = 0
    epsilon = 0.001
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, "model_" + str(max_game_number) + "_" + str(epsilon))
    model_directory = os.path.join(dir_path, "model.ckpt")

    def __init__(self, sess):
        self.sess = sess

        with tf.name_scope('value_network'):
            self.input = tf.placeholder(
                dtype=tf.float32, shape=(None, num_of_rows*num_of_cols*meta_data_cols), name='input')
            reshaped = tf.reshape(self.input, shape=[-1, num_of_rows, num_of_cols, meta_data_cols], name="reshape")
            c1 = tf.layers.conv2d(reshaped, filters=32, kernel_size=3, strides=2, activation=tf.nn.relu, name="conv1")
            flatten = tf.layers.flatten(c1, name="flatten")
            h1 = tf.layers.dense(flatten, 256, activation=tf.nn.relu)
            h2 = tf.layers.dense(h1, 128, activation=tf.nn.relu)
            self.output = tf.layers.dense(h2, 1, activation=tf.nn.sigmoid)
            self.foutput = tf.squeeze(self.output, axis=1, name='flatten_output')

        self.saver = tf.train.Saver()

        with tf.name_scope('training'):
            self.expected = tf.placeholder(
                dtype=tf.float32, shape=(None, 1), name='expected')
            cost = tf.reduce_mean((self.output - self.expected) ** 2)
            self.train = tf.train.AdamOptimizer(epsilon=NeuralDraught.epsilon).minimize(cost)

    def choose_board(self, possible_boards, player_moving):
        values = self.sess.run(self.foutput, feed_dict={
            self.input: possible_boards
        })
        values = self.consider_previous_choices(possible_boards, values)
        sum_of_values = sum(values)

        if sum_of_values < 0.001:
            idx = np.random.choice(range(len(possible_boards)))
        else:
            weights = values/sum_of_values
            idx = np.random.choice(range(len(possible_boards)), p=weights)

        board_str = ','.join(str(e) for e in possible_boards[idx])
        NeuralDraught.single_game_history.append({'PLAYER': player_moving, 'BOARD': board_str})
        return str(idx)


    def consider_previous_choices(self, possible_boards, values):
        num_of_uses = [0] * len(values)
        for i in range(len(values)):
            board_str = ','.join(str(e) for e in possible_boards[i])
            if board_str in NeuralDraught.training_history:
                num_of_uses[i] = NeuralDraught.training_history[board_str][1]
        sum_of_uses = sum(num_of_uses)
        if sum_of_uses > 0:
            for i in range(len(values)):
                values[i] += values[i] * num_of_uses[i]/sum_of_uses
        return values


    def process_end_of_game(self, winner):
        NeuralDraught.game_number += 1
        self.update_current_history(winner)
        print("Game:", NeuralDraught.game_number, "Training:", NeuralDraught.training_number, "The winner is:", winner)
        if NeuralDraught.game_number == NeuralDraught.max_game_number:
            NeuralDraught.game_number = 0
            NeuralDraught.training_number += 1
            self.train_network()

    def update_current_history(self, winner):
        for element in NeuralDraught.single_game_history:
            is_winner_value = 1 if winner == element['PLAYER'] else \
                              0  # not winner
            if element['BOARD'] not in NeuralDraught.training_history:
                NeuralDraught.training_history[element['BOARD']] = [is_winner_value, 1]
            else:
                current_element_value = NeuralDraught.training_history[element['BOARD']]
                current_element_value[0] += is_winner_value
                current_element_value[1] += 1
                NeuralDraught.training_history[element['BOARD']] = current_element_value
        NeuralDraught.single_game_history = []

    def train_network(self):
        states = []
        values = []
        for key, value in NeuralDraught.training_history.items():
            if value[1] >= (0.01 * NeuralDraught.max_game_number):
                state = [int(s) for s in key.split(',')]
                states.append(state)
                values.append([value[0] / value[1]])
                print(key, value)

        for x in range(100):
            self.sess.run(self.train, feed_dict={
                self.input: states,
                self.expected: values
            })

        NeuralDraught.training_history = {}

        #input("Press any key")

        save_path = self.saver.save(sess, NeuralDraught.model_directory)
        #print("Model saved in path: %s" % save_path)


row_translator = {
    'W': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7},
    'B': {0: 7, 1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1, 7: 0}
}
col_translator = {
    'W': {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3},
    'B': {0: 3, 1: 3, 2: 2, 3: 2, 4: 1, 5: 1, 6: 0, 7: 0}
}


def parse_board(player_moving, possible_board_data):
    possible_board = [0]*32*meta_data_cols
    pieces_with_location = possible_board_data.split(", ")
    for piece_with_location in pieces_with_location:
        location, piece = piece_with_location.split("=")
        col_str, row_str = location.split(",")
        col = col_translator[player_moving][int(col_str)]
        row = row_translator[player_moving][int(row_str)]
        what_piece = 0 if piece == "W,P" else \
                     1 if piece == "W,K" else \
                     2 if piece == "B,P" else \
                     3  # piece == "B,K"
        possible_board[row*num_of_cols*meta_data_cols + col*meta_data_cols + what_piece] = 1
    return possible_board


class MyTCPHandler(socketserver.StreamRequestHandler):
    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """

    operation = ""

    def handle(self):
        # self.request is the TCP socket connected to the client
        data = self.rfile.readline().strip().decode("utf-8")
        if MyTCPHandler.operation == "":
            MyTCPHandler.operation = data
            self.request.sendall(b'0')
        elif MyTCPHandler.operation == "GET_MOVE":
            MyTCPHandler.operation = ""
            player_moving = data[0]
            possible_boards_with_moves = data[4:-3].split("CM, CM")
            possible_boards = []
            for possible_board_with_moves in possible_boards_with_moves:
                possible_board_data = possible_board_with_moves.split("M, B")[1][1:-2]
                possible_board = parse_board(player_moving, possible_board_data)
                possible_boards.append(possible_board)
            agent_result = agent.choose_board(possible_boards, player_moving)
            self.request.sendall(agent_result.encode())
        elif MyTCPHandler.operation == "END_GAME":
            MyTCPHandler.operation = ""
            winner_is = "X"
            if "WINNER:" in data:
                winner_is = data[len("WINNER:")]
            elif "DRAW:" in data:
                data = data[len("DRAW:")+len("WV"):]
                white_value_raw, black_value_raw = data.split("BV")
                white_value = float(white_value_raw)
                black_value = float(black_value_raw)
                winner_is = "W" if white_value > black_value else \
                            "B" if black_value > white_value else \
                            "X"  # value == value
            #print("Winner is:", winner_is)
            agent.process_end_of_game(winner_is)
            self.request.sendall("READY".encode())


def is_folder_empty(folder):
    return [f for f in os.listdir(folder) if not f.startswith('.')] == []


if __name__ == "__main__":
    HOST, PORT = "192.168.0.103", 8080

    logging.getLogger().setLevel(logging.INFO)

    model_exists = os.path.isdir(NeuralDraught.dir_path)
    if not model_exists:
        os.makedirs(NeuralDraught.dir_path)

    with tf.Session() as sess:
        agent = NeuralDraught(sess)

        sess.run(tf.global_variables_initializer())
        print("Model initialized.")
        if not is_folder_empty(NeuralDraught.dir_path):
            agent.saver.restore(sess, NeuralDraught.model_directory)
            print("Model restored.")

        # Create the server, binding to localhost on port 9999
        server = socketserver.TCPServer((HOST, PORT), MyTCPHandler)

        # Activate the server; this will keep running until you
        # interrupt the program with Ctrl-C
        server.serve_forever()

