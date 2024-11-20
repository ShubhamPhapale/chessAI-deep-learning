import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import chess.pgn
import numpy as np

class EfficientChessDataset:
    def __init__(self, pgn_file):
        self.pgn_file = pgn_file
        self.games = self.load_games()
        self.move_mapping = {}

    def load_games(self):
        games = []
        with open(self.pgn_file, 'r') as pgn:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                games.append(game)
        return games

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
        
        return tensor

    def generate_training_batch(self, batch_size=32):
        board_states, move_labels = [], []
        
        for _ in range(batch_size):
            game = np.random.choice(self.games)
            board = game.board()
            
            for move in game.mainline_moves():
                board_tensor = self.board_to_tensor(board)
                board_states.append(board_tensor)
                
                from_square = move.from_square
                to_square = move.to_square
                move_index = from_square * 64 + to_square
                
                move_label = torch.zeros(20480)
                move_label[move_index] = 1
                move_labels.append(move_label)
                
                board.push(move)
        
        return torch.stack(board_states), torch.stack(move_labels)

class CompactChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(14, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 20480)  # Move scores

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(dataset, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompactChessNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        board_states, move_labels = dataset.generate_training_batch()
        board_states, move_labels = board_states.to(device), move_labels.to(device)

        optimizer.zero_grad()
        outputs = model(board_states)
        loss = criterion(outputs, move_labels)

        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Save the model
    torch.save(model.state_dict(), 'chess_model.pth')
    return model

class ChessMovePredictor:
    def __init__(self, model_path='chess_model.pth'):
        self.model = CompactChessNet()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

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
        
        return tensor.unsqueeze(0)

    def get_best_move(self, board):
        board_tensor = self.board_to_tensor(board)
        with torch.no_grad():
            move_scores = self.model(board_tensor).squeeze()
        
        legal_moves = list(board.legal_moves)
        move_candidates = []
        
        for move in legal_moves:
            from_square = move.from_square
            to_square = move.to_square
            move_index = from_square * 64 + to_square
            
            score = move_scores[move_index] if move_index < len(move_scores) else -float('inf')
            move_candidates.append((move, score))
        
        return max(move_candidates, key=lambda x: x[1])[0]

def play_game():
    predictor = ChessMovePredictor()
    board = chess.Board()
    
    while not board.is_game_over():
        move = predictor.get_best_move(board)
        board.push(move)
        print(f"Move: {move}")
        print(board)
        print("\n")

if __name__ == "__main__":
    # Training
    dataset = EfficientChessDataset('data.pgn')
    trained_model = train_model(dataset)
    
    # Play a game
    play_game()