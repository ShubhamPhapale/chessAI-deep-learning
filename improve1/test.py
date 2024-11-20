import torch
import chess
import logging
import os

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'chess_testing.log'), 
                    level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedChessNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(14, 64, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(256)
        
        self.residual_conv = torch.nn.Conv2d(14, 256, kernel_size=1)
        
        self.fc1 = torch.nn.Linear(256 * 8 * 8, 1024)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc3 = torch.nn.Linear(512, 20480)

    def forward(self, x):
        residual = self.residual_conv(x)
        
        x = self.bn1(self.conv1(x))
        x = torch.nn.functional.leaky_relu(x, 0.1)
        x = self.bn2(self.conv2(x))
        x = torch.nn.functional.leaky_relu(x, 0.1)
        x = self.bn3(self.conv3(x))
        x = torch.nn.functional.leaky_relu(x, 0.1)
        
        x += residual
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.nn.functional.leaky_relu(x, 0.1)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = torch.nn.functional.leaky_relu(x, 0.1)
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

class ChessMovePredictor:
    def __init__(self, model_path='checkpoints/chess_model_epoch_10.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EnhancedChessNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        logging.info(f"Model loaded from {model_path}")

    def board_to_tensor(self, board):
        tensor = torch.zeros(14, 8, 8)
        piece_order = [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                       chess.ROOK, chess.QUEEN, chess.KING]
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank, file = chess.square_rank(square), chess.square_file(square)
                channel = (piece_order.index(piece.piece_type) + 
                           (6 if piece.color == chess.BLACK else 0))
                tensor[channel, 7-rank, file] = 1
        
        tensor[12, :, :] = 1 if board.turn == chess.WHITE else 0
        tensor[13, :, :] = 1 if board.castling_rights else 0
        
        return tensor.unsqueeze(0).to(self.device)

    def get_best_move(self, board):
        board_tensor = self.board_to_tensor(board)
        with torch.no_grad():
            move_scores = self.model(board_tensor).squeeze().cpu()
        
        legal_moves = list(board.legal_moves)
        move_candidates = []
        
        for move in legal_moves:
            from_square = move.from_square
            to_square = move.to_square
            move_index = from_square * 64 + to_square
            
            score = move_scores[move_index] if move_index < len(move_scores) else -float('inf')
            move_candidates.append((move, score))
        
        best_move = max(move_candidates, key=lambda x: x[1])[0]
        logging.info(f"Best move predicted: {best_move}")
        return best_move

def test_ai_game():
    predictor = ChessMovePredictor()
    board = chess.Board()
    
    print("Starting Chess AI Game Test")
    logging.info("Starting Chess AI Game Test")
    
    move_count = 0
    max_moves = 300  # Prevent infinite game
    
    while not board.is_game_over() and move_count < max_moves:
        print(f"\nMove {move_count + 1}") #  - {board.turn_stack}"
        print(board)
        
        move = predictor.get_best_move(board)
        board.push(move)
        
        move_count += 1
    
    print("\nGame Finished:")
    print(board)
    print(f"Result: {board.result()}")
    print(f"Total Moves: {move_count}")

    logging.info(f"Game finished. Result: {board.result()}, Total Moves: {move_count}")

if __name__ == "__main__":
    test_ai_game()