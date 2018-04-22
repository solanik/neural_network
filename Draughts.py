import socketserver
import tensorflow as tf
import math
import numpy as np
import os.path

class NeuralDraught:
    single_game_history = []
    training_history = {}
    game_number = 0
    max_game_number = 1000
    training_number = 0
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, "model")
    model_directory = os.path.join(dir_path, "model.ckpt")

    def __init__(self, sess):
        self.sess = sess

        with tf.name_scope('value_network'):
            self.input = tf.placeholder(
                dtype=tf.float32, shape=(None,8*4*5), name='input')
            reshaped = tf.reshape(self.input, shape=[-1, 8, 4, 5], name="reshape")
            c1 = tf.layers.conv2d(reshaped, filters=32, kernel_size=3, strides=2, activation=tf.nn.relu, name="conv1")
            flatten = tf.layers.flatten(c1,name="flatten")
            h1 = tf.layers.dense(flatten, 512, activation=tf.nn.relu)
            h2 = tf.layers.dense(h1, 1024, activation=tf.nn.relu)
            h3 = tf.layers.dense(h2, 2048, activation=tf.nn.relu)
            self.output = tf.layers.dense(h3, 1, activation=tf.nn.sigmoid)
            self.foutput = tf.squeeze(self.output, axis=1, name='flatten_output')

        self.saver = tf.train.Saver()

        with tf.name_scope('training'):
            self.expected = tf.placeholder(
                dtype=tf.float32, shape=(None, 1), name='expected')
            cost = tf.reduce_mean((self.output - self.expected) ** 2)
            self.train = tf.train.AdamOptimizer(epsilon=0.01).minimize(cost)

    def choose_board(self, possible_boards, player_moving):
        values = self.sess.run(self.foutput, feed_dict={
            self.input: possible_boards
        })
        sum_of_values = sum(values)
        if (sum_of_values < 0.001):
            idx = np.random.choice(range(len(possible_boards)))
        else:
            weights = values/sum_of_values
            #print("Wartosci ktore otrzymalismy: ", weights)
            idx = np.random.choice(range(len(possible_boards)), p=weights)
        #print("Index ktory zwracamy: ", idx)
        board_str = ','.join(str(e) for e in possible_boards[idx])
        NeuralDraught.single_game_history.append({'PLAYER': player_moving, 'BOARD': board_str})
        return str(idx)

    def process_end_of_game(self, winner):
        NeuralDraught.game_number += 1
        self.update_current_history(winner)
        print("Game:", NeuralDraught.game_number)
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
            state = [int(s) for s in key.split(',')]
            states.append(state)
            values.append([value[0] / value[1]])

        self.sess.run(self.train, feed_dict={
            self.input: states,
            self.expected: values
        })

        NeuralDraught.training_history = {}

        save_path = self.saver.save(sess, NeuralDraught.model_directory)
        print("Model saved in path: %s" % NeuralDraught.model_directory)


def parse_board(player_moving, possible_board_data):
    possible_board = [0]*32*5
    pieces_with_location = possible_board_data.split(", ")
    for piece_with_location in pieces_with_location:
        location, piece = piece_with_location.split("=")
        col_str, row_str = location.split(",")
        col = math.floor(int(col_str)/2)
        row = int(row_str)
        what_piece = 0 if piece == "W,P" else \
                     1 if piece == "W,K" else \
                     2 if piece == "B,P" else \
                     3 # piece == "B,K"
        possible_board[row*20 + col*5 + what_piece] = 1
    player_moving_int = 1 if player_moving == "W" else \
                        0 # player_moving == "B"
    for i in range(32):
        possible_board[i*5 + 4] = player_moving_int
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
            #print("Received,", MyTCPHandler.operation)
            self.request.sendall(b'0')
        elif MyTCPHandler.operation == "GET_MOVE":
            MyTCPHandler.operation = ""
            #print("Parsing boards,", data)
            player_moving = data[0]
            possible_boards_with_moves = data[4:-3].split("CM, CM")
            possible_boards = []
            for possible_board_with_moves in possible_boards_with_moves:
                possible_board_data = possible_board_with_moves.split("M, B")[1][1:-2]
                possible_board = parse_board(player_moving, possible_board_data)
                #print("Possible board: ", possible_board)
                possible_boards.append(possible_board)
            agentResult = agent.choose_board(possible_boards, player_moving)
            self.request.sendall(agentResult.encode())
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
                #print("Game ended with result: ", winner_is)
                #input("Press Enter to continue...")
            #print("Game ended with result: ", winner_is)
            agent.process_end_of_game(winner_is)
            self.request.sendall("READY".encode())
        #print("END OF HANDLE")

if __name__ == "__main__":
    HOST, PORT = "192.168.0.103", 8080

    if not os.path.isdir(NeuralDraught.dir_path):
        os.makedirs(NeuralDraught.dir_path)

    with tf.Session() as sess:
        agent = NeuralDraught(sess)

        if os.path.isfile(NeuralDraught.model_directory):
            agent.saver.restore(sess, NeuralDraught.model_directory)
            print("Model restored.")
        else:
            sess.run(tf.global_variables_initializer())

        # Create the server, binding to localhost on port 9999
        server = socketserver.TCPServer((HOST, PORT), MyTCPHandler)

        # Activate the server; this will keep running until you
        # interrupt the program with Ctrl-C
        server.serve_forever()

