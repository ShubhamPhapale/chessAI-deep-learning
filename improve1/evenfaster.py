import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import chess.pgn
import numpy as np
from torch.utils.data import Dataset, DataLoader, IterableDataset
import logging
import os
from torch.optim.lr_scheduler import StepLR
from typing import Iterator, Optional, Tuple
import mmap
import gc
from collections import deque
import psutil
import io

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'chess_training.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ChessGameIterator:
    def __init__(self, pgn_file: str):
        if not os.path.exists(pgn_file):
            raise FileNotFoundError(f"PGN file not found: {pgn_file}")
        self.pgn_file = pgn_file
        self.file_size = os.path.getsize(pgn_file)
        self._init_mmap()

    def _init_mmap(self):
        try:
            self.file = open(self.pgn_file, 'rb')
            self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception as e:
            logging.error(f"Error initializing mmap for file {self.pgn_file}: {str(e)}")
            raise

    def __iter__(self):
        self.mmap.seek(0)
        return self

    def __next__(self):
        if self.mmap.tell() >= self.file_size:
            self.mmap.seek(0)
            raise StopIteration
            
        game_str = ""
        try:
            while True:
                line = self.mmap.readline().decode('utf-8')
                if not line:
                    break
                game_str += line
                if line.strip() == "":
                    if game_str.strip():
                        string_io = io.StringIO(game_str)
                        game = chess.pgn.read_game(string_io)
                        if game is not None:
                            return game
                    game_str = ""
            
            if game_str.strip():
                string_io = io.StringIO(game_str)
                game = chess.pgn.read_game(string_io)
                if game is not None:
                    return game
                
        except Exception as e:
            logging.error(f"Error reading game: {str(e)}")
            raise StopIteration
            
        raise StopIteration

    def close(self):
        try:
            self.mmap.close()
            self.file.close()
        except Exception as e:
            logging.error(f"Error closing files: {str(e)}")

    def __del__(self):
        self.close()

class EfficientChessDataset(IterableDataset):
    def __init__(self, pgn_file: str, cache_size: int = 1000):
        self.pgn_file = pgn_file
        self.cache_size = cache_size
        self.cache = deque(maxlen=cache_size)
        self.game_iterator = None
        logging.info(f"Initializing dataset from {pgn_file} with cache size {cache_size}")

    def _fill_cache(self):
        try:
            if not self.game_iterator:
                self.game_iterator = ChessGameIterator(self.pgn_file)
            
            while len(self.cache) < self.cache_size:
                try:
                    game = next(self.game_iterator)
                    if game:
                        self.cache.append(game)
                except StopIteration:
                    break
        except Exception as e:
            logging.error(f"Error filling cache: {str(e)}")
            raise

    def board_to_tensor(self, board) -> torch.Tensor:
        tensor = torch.zeros(14, 8, 8, dtype=torch.float16)
        piece_order = [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                      chess.ROOK, chess.QUEEN, chess.KING]
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank, file = chess.square_rank(square), chess.square_file(square)
                channel = (piece_order.index(piece.piece_type) + 
                          (6 if piece.color == chess.BLACK else 0))
                tensor[channel, 7-rank, file] = 1
        
        tensor[12, :, :] = float(board.turn == chess.WHITE)
        tensor[13, :, :] = float(bool(board.castling_rights))
        
        return tensor

    def __iter__(self) -> Iterator:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            self.game_iterator = ChessGameIterator(self.pgn_file)
            for _ in range(worker_id):
                next(self.game_iterator)
        else:
            self.game_iterator = ChessGameIterator(self.pgn_file)

        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.cache) < self.cache_size // 2:
            self._fill_cache()
        
        if not self.cache:
            raise StopIteration

        try:
            game = self.cache.popleft()
            board = game.board()
            
            board_tensor = self.board_to_tensor(board)
            
            mainline_moves = list(game.mainline_moves())
            if mainline_moves:
                first_move = mainline_moves[0]
                from_square = first_move.from_square
                to_square = first_move.to_square
                move_index = from_square * 64 + to_square
                
                move_label = torch.zeros(20480, dtype=torch.float16)
                move_label[move_index] = 1
            else:
                move_label = torch.zeros(20480, dtype=torch.float16)
            
            if psutil.Process().memory_percent() > 80:
                gc.collect()
                torch.cuda.empty_cache()
            
            return board_tensor, move_label
        except Exception as e:
            logging.error(f"Error processing game: {str(e)}")
            raise

    def __del__(self):
        if self.game_iterator:
            self.game_iterator.close()

class EnhancedChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(14, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Residual connection
        self.residual_conv = nn.Conv2d(14, 256, kernel_size=1)
        
        # Dense layers with dropout
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 20480)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection
        residual = self.residual_conv(x)
        
        # Convolutional layers
        x = self.bn1(self.conv1(x))
        x = F.leaky_relu(x, 0.1)
        x = self.bn2(self.conv2(x))
        x = F.leaky_relu(x, 0.1)
        x = self.bn3(self.conv3(x))
        x = F.leaky_relu(x, 0.1)
        
        # Add residual connection
        x += residual
        
        # Dense layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.leaky_relu(x, 0.1)
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

class MemoryEfficientTrainer:
    def __init__(self, model: nn.Module, criterion: nn.Module, 
                 optimizer: optim.Optimizer, scheduler: Optional[optim.lr_scheduler._LRScheduler] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = torch.amp.GradScaler()

    @torch.amp.autocast(device_type='cuda')  # Updated this line
    def train_step(self, board_states: torch.Tensor, move_labels: torch.Tensor) -> float:
        outputs = self.model(board_states)
        loss = self.criterion(outputs, move_labels)
        return loss

    def train_epoch(self, dataloader: DataLoader, accumulation_steps: int = 1) -> float:
        self.model.train()
        running_loss = 0.0
        
        for step, (board_states, move_labels) in enumerate(dataloader):
            # Prefetch next batch
            if hasattr(dataloader, 'prefetch_hook'):
                dataloader.prefetch_hook()
            
            board_states = board_states.to(self.device, non_blocking=True)
            move_labels = move_labels.to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.model(board_states)
                loss = self.criterion(outputs, move_labels)
                loss = loss / accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()

            # Less frequent memory cleanup
            if step % 500 == 0:
                del board_states, move_labels, outputs, loss
                torch.cuda.empty_cache()

        if self.scheduler:
            self.scheduler.step()

        return running_loss / len(dataloader)


def create_dataloader(pgn_file: str, batch_size: int = 128, num_workers: int = 8, 
                     cache_size: int = 2000) -> DataLoader:
    dataset = EfficientChessDataset(pgn_file, cache_size=cache_size)
    
    def collate_fn(batch):
        board_states = torch.stack([item[0] for item in batch])
        move_labels = torch.stack([item[1] for item in batch])
        return board_states, move_labels

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

class ChessMovePredictor:
    def __init__(self, model_path: str = 'chess_model.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EnhancedChessNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        logging.info(f"Model loaded from {model_path}")

    def board_to_tensor(self, board) -> torch.Tensor:
        tensor = torch.zeros(14, 8, 8, dtype=torch.float16)
        piece_order = [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                      chess.ROOK, chess.QUEEN, chess.KING]
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank, file = chess.square_rank(square), chess.square_file(square)
                channel = (piece_order.index(piece.piece_type) + 
                          (6 if piece.color == chess.BLACK else 0))
                tensor[channel, 7-rank, file] = 1
        
        tensor[12, :, :] = float(board.turn == chess.WHITE)
        tensor[13, :, :] = float(bool(board.castling_rights))
        
        return tensor.unsqueeze(0).to(self.device)

    def get_best_move(self, board) -> chess.Move:
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

def train_model(pgn_file: str, model: nn.Module, epochs: int = 200, 
                batch_size: int = 128, num_workers: int = 8, cache_size: int = 2000):
    logging.info("Initializing training components...")
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Pin memory and enable prefetch
    dataloader = create_dataloader(
        pgn_file, 
        batch_size=batch_size,
        num_workers=num_workers,
        cache_size=cache_size
    )
    
    trainer = MemoryEfficientTrainer(model, criterion, optimizer, scheduler)
    
    # Enable cudnn benchmarking for better performance
    torch.backends.cudnn.benchmark = True
    
    logging.info("Starting training...")
    for epoch in range(epochs):
        epoch_loss = trainer.train_epoch(dataloader, accumulation_steps=1)  # Reduced accumulation steps
        logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, f'checkpoints/chess_model_epoch_{epoch+1}.pth')
            
    torch.save(model.state_dict(), 'chess_model_final.pth')
    logging.info("Training completed.")

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
    # Configuration with optimized parameters
    pgn_file = '../data.pgn'
    config = {
        'epochs': 200,
        'batch_size': 1024,  # Increased batch size
        'num_workers': 16,   # Increased workers
        'cache_size': 1000, # Increased cache size
        'mode': 'train'
    }

    try:
        model = EnhancedChessNet()
        
        if config['mode'] == 'train':
            logging.info(f"Starting training with config: {config}")
            
            if torch.cuda.is_available():
                logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                # Set GPU settings for better performance
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            else:
                logging.warning("CUDA not available, using CPU for training")
            
            train_model(
                pgn_file=pgn_file,
                model=model,
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                num_workers=config['num_workers'],
                cache_size=config['cache_size']
            )
            
            logging.info("Training completed successfully")
            
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        torch.save(model.state_dict(), 'chess_model_interrupted.pth')
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        raise
        
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleanup completed")