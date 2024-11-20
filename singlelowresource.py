import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import chess.pgn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import logging
import os

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'chess_training.log'), 
                    level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class EfficientChessDataset(Dataset):
    def __init__(self, pgn_file, max_games=None):
        """
        Initialize the dataset with a PGN file and an optional max_games argument.
        
        Args:
            pgn_file (str): Path to the PGN file containing chess games.
            max_games (int, optional): Maximum number of games to load. Defaults to None, which loads all games.
        """
        self.pgn_file = pgn_file
        logging.info(f"Loading games from {pgn_file}...")
        self.max_games = max_games
        self.games = self.load_games()

    def load_games(self):
        games = []
        with open(self.pgn_file, 'r') as pgn:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                games.append(game)
                if self.max_games and len(games) >= self.max_games:
                    break
        logging.info(f"Loaded {len(games)} games from the dataset.")
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

    def __getitem__(self, index):
        game = self.games[index]
        board = game.board()
        
        # For the first move, return the initial board state and first move label
        board_tensor = self.board_to_tensor(board)
        
        # Get the first move
        mainline_moves = list(game.mainline_moves())
        if mainline_moves:
            first_move = mainline_moves[0]
            from_square = first_move.from_square
            to_square = first_move.to_square
            move_index = from_square * 64 + to_square
            
            move_label = torch.zeros(20480)
            move_label[move_index] = 1
        else:
            move_label = torch.zeros(20480)
        
        return board_tensor, move_label

    def collate_fn(self, batch):
        board_states = torch.stack([item[0] for item in batch])
        move_labels = torch.stack([item[1] for item in batch])
    
        return board_states, move_labels

    def __len__(self):
        return len(self.games)


class CompactChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(14, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)  # Reduced size
        self.fc2 = nn.Linear(256, 20480)  # Move scores

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(dataloader, model, criterion, optimizer, epochs=100, accumulation_steps=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.train()
    logging.info("Training started...")
    for epoch in range(epochs):
        running_loss = 0.0
        for step, (board_states, move_labels) in enumerate(dataloader):
            board_states, move_labels = board_states.to(device), move_labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(board_states)
            loss = criterion(outputs, move_labels)

            loss.backward()

            # Gradient accumulation
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
        
        # Periodically save the model and log
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'checkpoints/chess_model_epoch_{epoch+1}.pth')
            logging.info(f"Model checkpoint saved for epoch {epoch+1}.")

    # Save the final model after training
    torch.save(model.state_dict(), 'chess_model.pth')
    logging.info("Training completed. Final model saved.")


class ChessMovePredictor:
    def __init__(self, model_path='chess_model.pth'):
        self.model = CompactChessNet()
        self.model.load_state_dict(torch.load(model_path))
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
        
        best_move = max(move_candidates, key=lambda x: x[1])[0]
        logging.info(f"Best move predicted: {best_move}")
        return best_move


def play_game():
    predictor = ChessMovePredictor()
    board = chess.Board()
    
    logging.info("Starting a new game...")
    while not board.is_game_over():
        move = predictor.get_best_move(board)
        board.push(move)
        logging.info(f"Move: {move}")
        logging.debug(f"Board after move:\n{board}")
        
        print(board)
        print("\n")
    
    logging.info("Game over.")
    print("Game over.")
    print(board.result())


if __name__ == "__main__":
    # Prepare dataset and DataLoader
    logging.info("Preparing dataset and DataLoader...")
    dataset = EfficientChessDataset('data.pgn', max_games=10000)  # Load up to 100 games by default
    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=dataset.collate_fn)
    logging.info("Dataset and DataLoader prepared.")

    # Initialize model, criterion, and optimizer
    logging.info("Initializing model, criterion, and optimizer...")
    model = CompactChessNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    logging.info("Model, criterion, and optimizer initialized.")

    # Train model
    train_model(dataloader, model, criterion, optimizer, epochs=100, accumulation_steps=4)

    # Play a game after training
    play_game()