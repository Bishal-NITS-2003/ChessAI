from fastapi import FastAPI
import torch
import torch.nn as nn
import chess


def get_sorted_indices(tensor):
    """Returns indices sorted by descending probability."""
    return torch.argsort(tensor, descending=True).tolist()



class ChessMovePredictor(nn.Module):
    def __init__(self):
        super(ChessMovePredictor, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)  # 64 for start square, 64 for end square
        self.fc3 = nn.Linear(128, 64)   # Start square
        self.fc4 = nn.Linear(128, 64)   # End square

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        start_square = self.fc3(x)  # Predict starting square
        end_square = self.fc4(x)    # Predict ending square
        return start_square, end_square
    
def fen_to_tensor(fen):
    board = chess.Board(fen)
    tensor = torch.zeros((12, 8, 8))

    piece_map = {
        'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
        'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11
    }

    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        tensor[piece_map[piece.symbol()], row, col] = 1

    return tensor    

# Define API app
app = FastAPI()

all_moves = list(chess.SQUARE_NAMES)  # 'a1', 'a2', ..., 'h8'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = ChessMovePredictor()  # Initialize model
model.load_state_dict(torch.load("chess_model.pth", map_location=torch.device(device)))  # Load on CPU
model.to(device)  # Ensure it's on CPU
model.eval()  # Set to evaluation mode

@app.get("/")
def home():
    return {"message": "Model API is running!"}

@app.post("/predict/")
async def predict(fen: str):
    try:
        board = chess.Board(fen)
        # Run model prediction
        with torch.no_grad():
            tensor = fen_to_tensor(fen).unsqueeze(0).to(device)  # Add batch dimension
            start_probs, end_probs = model(tensor)
   
             # Get sorted move candidates based on probability
            start_candidates = get_sorted_indices(start_probs.squeeze())
            end_candidates = get_sorted_indices(end_probs.squeeze())

            # Find the best legal move
            for start_idx in start_candidates:
                  for end_idx in end_candidates:
                       start_square = all_moves[start_idx]
                       end_square = all_moves[end_idx]
                       move = chess.Move.from_uci(f"{start_square}{end_square}")

                       if move in board.legal_moves:
                               return {"prediction": move.uci()}

            return {"prediction": "No move found"}

    except Exception as e:
        return {"error": str(e)}

