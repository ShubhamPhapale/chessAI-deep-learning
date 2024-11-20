import torch
import chess
import numpy as np

from train import CompactChessNet

class ChessMovePredictor:
    def __init__(self, model_path='model_quantized.pt'):
        # Recreate the original model architecture
        self.model = CompactChessNet()
        
        # Load the full model state dict before quantization
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        
        # Ensure the model is in evaluation mode
        self.model.eval()
        
    def board_to_tensor(self, board):
        """
        Convert a chess board to a 14-channel tensor representation
        
        Args:
            board (chess.Board): Current chess board state
        
        Returns:
            torch.Tensor: 14x8x8 tensor representation of the board
        """
        # 14 channels: 6 pieces for white, 6 for black, 2 for additional info
        tensor = torch.zeros(14, 8, 8)
        
        # Piece channels (0-5 white, 6-11 black)
        piece_order = [
            chess.PAWN, chess.KNIGHT, chess.BISHOP, 
            chess.ROOK, chess.QUEEN, chess.KING
        ]
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank, file = chess.square_rank(square), chess.square_file(square)
                channel = (piece_order.index(piece.piece_type) + 
                           (6 if piece.color == chess.BLACK else 0))
                tensor[channel, 7-rank, file] = 1
        
        # Additional channels (e.g., turn, castling rights)
        tensor[12, :, :] = 1 if board.turn == chess.WHITE else 0
        tensor[13, :, :] = 1 if board.castling_rights else 0
        
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def get_best_move(self, board):
        """
        Generate the best move for the current board state
        
        Args:
            board (chess.Board): Current chess board state
        
        Returns:
            chess.Move: Predicted best move
        """
        # Convert board to input tensor
        board_tensor = self.board_to_tensor(board)
        
        # Get move scores from the model
        with torch.no_grad():
            move_scores = self.model(board_tensor).squeeze()
        
        # Get legal moves
        legal_moves = list(board.legal_moves)
        
        # Prepare move representation matching your model's output
        all_moves = []
        for move in legal_moves:
            # Convert move to your model's move representation 
            # This is a placeholder and might need adjustment based on your exact encoding
            from_square = move.from_square
            to_square = move.to_square
            
            # Create a unique index for the move
            move_index = from_square * 64 + to_square
            
            # Get the score for this move
            score = move_scores[move_index] if move_index < len(move_scores) else -float('inf')
            all_moves.append((move, score))
        
        # Select the move with the highest score
        best_move = max(all_moves, key=lambda x: x[1])[0]
        return best_move

# Example usage
def play_game():
    predictor = ChessMovePredictor()
    board = chess.Board()
    
    while not board.is_game_over():
        print("shubh")
        # Get best move for current player
        move = predictor.get_best_move(board)
        print("labh")
        # Make the move
        board.push(move)
        print(f"Move played: {move}")
        print(board)

if __name__ == "__main__":
    play_game()