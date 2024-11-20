import chess.pgn
import numpy as np
import torch
import random

class EfficientChessDataset:
    def __init__(self, pgn_file, max_games=50000, sample_size=10000):
        self.pgn_file = pgn_file
        self.max_games = max_games
        self.sample_size = sample_size
        self.game_moves = self.extract_game_moves()

    def extract_game_moves(self):
        game_moves = []
        with open(self.pgn_file, 'r') as pgn:
            for _ in range(self.max_games):
                game = chess.pgn.read_game(pgn)
                if not game:
                    break

                # Skip games with very few moves
                moves = list(game.mainline_moves())
                if moves and len(moves) > 10:
                    game_moves.append(moves)

        return game_moves

    # def generate_training_batch(self):
    #     # Randomly sample games and moves
    #     sampled_moves = random.sample(self.game_moves, min(len(self.game_moves), self.sample_size))

    #     boards = []
    #     moves = []

    #     for game_moves in sampled_moves:
    #         board = chess.Board()
    #         for move in game_moves:
    #             # Encode board state
    #             boards.append(self.encode_board(board))

    #             # Encode move
    #             moves.append(self.encode_move(move))

    #             board.push(move)

    #     return (torch.stack(boards), torch.stack(moves))

    def generate_training_batch(self, batch_size=8):
        # Randomly sample games and moves
        sampled_moves = random.sample(self.game_moves, min(len(self.game_moves), self.sample_size))

        boards = []
        moves = []

        for game_moves in sampled_moves:
            board = chess.Board()
            for move in game_moves:
                # Encode board state
                boards.append(self.encode_board(board))

                # Encode move
                moves.append(self.encode_move(move))

                board.push(move)

        # Convert to tensors
        boards_tensor = torch.stack(boards)
        moves_tensor = torch.stack(moves)

        # Divide the data into batches
        num_batches = len(boards_tensor) // batch_size
        if len(boards_tensor) % batch_size != 0:
            num_batches += 1

        # Yield batches
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(boards_tensor))
            yield boards_tensor[start_idx:end_idx], moves_tensor[start_idx:end_idx]

    def encode_board(self, board):
        encoding = np.zeros((14, 8, 8), dtype=np.float32)

        piece_mapping = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                color_offset = 6 if piece.color == chess.BLACK else 0
                channel = piece_mapping[piece.piece_type] + color_offset
                encoding[channel][row][col] = 1

        encoding[12][0][0] = 1 if board.turn == chess.WHITE else 0
        encoding[13][0][0] = board.castling_rights

        return torch.tensor(encoding)

    def encode_move(self, move):
        # Create a more compact move representation
        from_square = move.from_square
        to_square = move.to_square
        promotion = move.promotion if move.promotion else 0

        # Use a more compact encoding scheme
        # Represents: from_square (6 bits), to_square (6 bits), promotion (3 bits)
        move_encoding = np.zeros(64 * 64 * 5, dtype=np.float32)
        move_index = (from_square * 64 + to_square) * 5 + (promotion or 0)
        move_encoding[move_index] = 1

        return torch.tensor(move_encoding)