import os
import torch
from transformers import pipeline
from fastapi import FastAPI
import uvicorn
import nest_asyncio
from pyngrok import ngrok
from dotenv import load_dotenv

# .env読み込み
load_dotenv()

# ngrok認証
NGROK_TOKEN = os.environ["NGROK_TOKEN"]
ngrok.set_auth_token(NGROK_TOKEN)

# モデル設定
MODEL_NAME = "google/gemma-2-2b-jpn-it"  # ←ここがガチポイント

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻 使用デバイス: {device}")

# モデルロード
print("🌀 モデル読み込み中...")
model = pipeline(
    "text-generation",
    model=MODEL_NAME,
    device=0 if device == "cuda" else -1,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
)
print("✅ モデル読み込み完了！")

# FastAPI app作成
app = FastAPI()

# ルート
@app.get("/")
def read_root():
    return {"message": "Gemma 2B API server is running!"}

# 推論エンドポイント
@app.post("/predict")
def predict(data: dict):
    message = data.get("message", "")
    print(f"💬 受け取ったメッセージ: {message}")
    output = model(message, max_new_tokens=100)
    generated_text = output[0]["generated_text"]
    return {"response": generated_text}

# 公開URL表示＆サーバー起動
public_url = ngrok.connect(8000)
print(f"✨ API公開URL: {public_url}")
nest_asyncio.apply()
uvicorn.run(app, host="0.0.0.0", port=8000)
