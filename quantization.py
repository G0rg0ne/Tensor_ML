import torch
import torch.nn as nn
from torchao.quantization import quantize_
from torchao.quantization.quant_api import Int8WeightOnlyConfig, Int4WeightOnlyConfig
import copy
import os

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=512, num_heads=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc_out(x)

def get_model_size_mb(model):
    """Calculate model size in MegaBytes."""
    # We use state_dict to simulate disk size / parameter memory
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / (1024 * 1024)
    os.remove("temp.p")
    return size

def check_fbgemm_genai():
    """Check if fbgemm-gpu-genai is available for int4 quantization."""
    try:
        import fbgemm_gpu_genai
        return True
    except ImportError:
        return False


device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = MiniTransformer().to(device).to(torch.bfloat16)


model_int8 = copy.deepcopy(base_model)
model_int4 = copy.deepcopy(base_model)


try:
    quantize_(model_int8, Int8WeightOnlyConfig())
    int8_success = True
except Exception as e:
    print(f"⚠️  INT8 quantization failed: {e}")
    int8_success = False

int4_success = False
if check_fbgemm_genai():
    try:
        quantize_(model_int4, Int4WeightOnlyConfig(group_size=32))
        int4_success = True
    except ImportError as e:
        print(f"⚠️  INT4 quantization failed: {e}")
        print("   Install with: pip install fbgemm-gpu-genai>=1.2.0")
    except Exception as e:
        print(f"⚠️  INT4 quantization failed: {e}")
        int4_success = False
else:
    print("⚠️  INT4 quantization requires fbgemm-gpu-genai >= 1.2.0")
    print("   Install with: pip install fbgemm-gpu-genai>=1.2.0")

print(f"\n{'Quantization Level':<20} | {'Memory (MB)':<15}")
print("-" * 40)
print(f"{'BF16 (Original)':<20} | {get_model_size_mb(base_model):>10.2f} MB")

if int8_success:
    print(f"{'INT8 (Weight-Only)':<20} | {get_model_size_mb(model_int8):>10.2f} MB")
else:
    print(f"{'INT8 (Weight-Only)':<20} | {'N/A (Failed)':>15}")

if int4_success:
    print(f"{'INT4 (Weight-Only)':<20} | {get_model_size_mb(model_int4):>10.2f} MB")
else:
    print(f"{'INT4 (Weight-Only)':<20} | {'N/A (Missing deps)':>15}")