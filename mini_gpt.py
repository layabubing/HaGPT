import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time
from datetime import datetime

# =============================
# 超参数
# =============================
batch_size = 32
block_size = 128
max_iters = 3000
eval_interval = 300
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

n_embd = 256
n_head = 8
n_layer = 6
dropout = 0.1


# =============================
# 数据加载 (tiny Shakespeare)
# =============================
print("=== 开始加载数据 ===")
print(f"当前工作目录: {os.getcwd()}")
print(f"输入文件路径: {os.path.abspath('input.txt')}")

try:
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    print(f"成功加载数据，文本长度: {len(text)} 字符")
    print(f"前100个字符: {text[:100]}...")
except FileNotFoundError:
    print("错误: input.txt 文件未找到!")
    print("请确保input.txt文件存在于当前目录")
    # 创建示例文本用于测试
    text = "Hello world! This is a test text for mini GPT" * 100
    print(f"使用测试文本，长度: {len(text)} 字符")

chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f"词汇表大小: {vocab_size}")
print(f"字符集: {''.join(chars[:50])}..." if len(chars) > 50 else f"字符集: {''.join(chars)}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

print("字符到索引映射示例:")
for i, char in enumerate(chars[:10]):
    print(f"  '{char}' -> {stoi[char]}")

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    
    # 确保有足够的数据
    if len(data) <= block_size:
        raise ValueError(f"数据长度({len(data)})小于block_size({block_size})")
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    # 调试信息
    if split == "train" and torch.randint(0, 100, (1,)).item() < 5:  # 5%概率打印调试信息
        print(f"  Batch形状: x={x.shape}, y={y.shape}")
        print(f"  样本内容: {decode(x[0][:20].tolist())}")
    
    return x.to(device), y.to(device)


# =============================
# Attention 头
# =============================
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        
        # 调试信息
        if torch.randint(0, 1000, (1,)).item() < 1:  # 0.1%概率打印
            print(f"    Head输入形状: {x.shape}")

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        
        # 调试信息
        if torch.randint(0, 1000, (1,)).item() < 1:
            print(f"    Head输出形状: {out.shape}")
            print(f"    注意力权重形状: {wei.shape}")
        
        return out


# =============================
# Multi-Head Attention
# =============================
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)


# =============================
# 前馈网络
# =============================
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# =============================
# Transformer Block
# =============================
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention()
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# =============================
# GPT 模型
# =============================
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        print(f"=== GPT模型初始化 ===")
        print(f"词汇表大小: {vocab_size}")
        print(f"嵌入维度: {n_embd}")
        print(f"层数: {n_layer}")
        print(f"注意力头数: {n_head}")

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # 调试信息
        if torch.randint(0, 100, (1,)).item() < 5:
            print(f"  GPT输入形状: {idx.shape}")
            print(f"  输入内容示例: {decode(idx[0][:10].tolist())}")

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        
        if torch.randint(0, 100, (1,)).item() < 5:
            print(f"  嵌入后形状: {x.shape}")

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if torch.randint(0, 100, (1,)).item() < 5:
            print(f"  输出logits形状: {logits.shape}")

        if targets is None:
            loss = None
        else:
            logits = logits.view(B*T, vocab_size)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
            if torch.randint(0, 100, (1,)).item() < 5:
                print(f"  损失值: {loss.item():.4f}")

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


print(f"=== 开始训练 ===")
print(f"设备: {device}")
print(f"最大迭代次数: {max_iters}")
print(f"学习率: {learning_rate}")

model = GPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 打印模型参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数总数: {total_params:,}")

# =============================
# 训练
# =============================
start_time = time.time()

for iter in range(max_iters):
    model.train()
    
    try:
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()

        if iter % eval_interval == 0:
            elapsed_time = time.time() - start_time
            print(f"step {iter}: loss {loss.item():.4f}, 耗时: {elapsed_time:.2f}s")
            
            # 验证集评估
            model.eval()
            with torch.no_grad():
                val_xb, val_yb = get_batch("val")
                val_logits, val_loss = model(val_xb, val_yb)
                print(f"  验证集损失: {val_loss.item():.4f}")
    
    except Exception as e:
        print(f"训练步骤 {iter} 出错: {e}")
        break

print(f"训练完成，总耗时: {time.time() - start_time:.2f}s")


# =============================
# 生成文本
# =============================
print("\n=== 开始生成文本 ===")
model.eval()

def generate_from_prompt(prompt, max_tokens=300, temperature=1.0):
    """根据用户输入的prompt生成文本"""
    # 将prompt编码为token
    prompt_tokens = encode(prompt)
    
    # 如果prompt太长，截取到block_size
    if len(prompt_tokens) > block_size:
        prompt_tokens = prompt_tokens[-block_size:]
        print(f"提示词过长，已截取后{block_size}个token")
    
    # 转换为tensor并移动到设备
    context = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    print(f"\n--- 生成结果 ---")
    print(f"提示词: '{prompt}'")
    print(f"提示词长度: {len(prompt_tokens)} tokens")
    
    with torch.no_grad():
        # 生成文本
        generated = model.generate(context, max_new_tokens=max_tokens)
        generated_text = decode(generated[0].tolist())
        
        # 显示结果（保留原始prompt）
        print(f"生成文本: {generated_text}")
        
        # 只显示新生成的部分
        new_text = generated_text[len(prompt):]
        print(f"新生成部分: {new_text}")
    
    return generated_text

# 交互式prompt输入
print("\n=== 交互式文本生成 ===")
print("输入 'quit' 或 'exit' 退出程序")
print("输入 'help' 查看可用命令")

while True:
    try:
        user_input = input("\n请输入prompt: ").strip()
        
        if user_input.lower() in ['quit', 'exit', '退出']:
            print("感谢使用mini GPT!")
            break
        
        elif user_input.lower() in ['help', '帮助']:
            print("""
可用命令:
- 输入任意文本作为prompt进行生成
- 'quit'/'exit'/'退出': 退出程序
- 'help'/'帮助': 显示此帮助信息
- 'example': 查看示例prompt
- 'length <数字>': 设置生成长度（默认300）
            """)
        
        elif user_input.lower() == 'example':
            print("""
示例prompt:
- "The city was quiet"
- "Ethan walked slowly"
- "Language itself seemed"
- "Winter arrived quietly"
- "Spring returned"
            """)
        
        elif user_input.lower().startswith('length '):
            try:
                new_length = int(user_input.split()[1])
                max_tokens = new_length
                print(f"生成长度设置为: {max_tokens}")
            except:
                print("格式错误，请使用: length <数字>")
        
        elif user_input:
            # 检查prompt中的字符是否在词汇表中
            unknown_chars = [c for c in user_input if c not in stoi]
            if unknown_chars:
                print(f"警告: 以下字符不在词汇表中: {set(unknown_chars)}")
                print("这些字符将被忽略或替换")
            
            # 生成文本
            generate_from_prompt(user_input, max_tokens)
        
        else:
            print("请输入有效的prompt")
    
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        break
    except Exception as e:
        print(f"生成过程中出错: {e}")

print("\n=== 程序执行完成 ===")