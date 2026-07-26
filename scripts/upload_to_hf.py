"""
Automated Hugging Face Publisher for GeoTree Model
Uploads model weights, config, and Model Card to Hugging Face Model Hub (repo: geotree).
"""
import os
import sys
import shutil
from pathlib import Path

def publish():
    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("Installing huggingface_hub package...")
        os.system("pip install huggingface_hub")
        from huggingface_hub import HfApi, login

    print("=" * 60)
    print(" 🌳 GeoTree Hugging Face Model Hub Publisher")
    print("=" * 60)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    token = os.getenv("HF_TOKEN")
    if not token:
        token = input("Enter your Hugging Face Access Token (or set HF_TOKEN env var): ").strip()

    if not token:
        print("❌ Error: Hugging Face token is required to publish.")
        sys.exit(1)

    login(token=token)
    api = HfApi()

    # Prepare model directory
    weights_src = Path("weights/best_model.pth")
    hf_dir = Path("huggingface")
    weights_dst = hf_dir / "pytorch_model.bin"

    if weights_src.exists():
        shutil.copy(weights_src, weights_dst)
        print(f"✅ Copied model weights from {weights_src} -> {weights_dst}")
    else:
        print(f"❌ Error: Weights file not found at {weights_src}")
        sys.exit(1)

    repo_name = "geotree"
    print(f"\nPublishing repository to Hugging Face Model Hub as: {repo_name}...")

    try:
        user_info = api.whoami()
        username = user_info["name"]
        full_repo_id = f"{username}/{repo_name}"

        # Create repo if not exists
        api.create_repo(repo_id=full_repo_id, exist_ok=True, repo_type="model")
        print(f"✅ Repository confirmed/created: https://huggingface.co/{full_repo_id}")

        # Upload folder
        api.upload_folder(
            folder_path=str(hf_dir),
            repo_id=full_repo_id,
            repo_type="model",
            commit_message="Initial release of GeoTree model weights & model card"
        )

        print("\n" + "=" * 60)
        print(" 🎉 SUCCESS! GeoTree model is now published on Hugging Face!")
        print(f" 🔗 Model Link: https://huggingface.co/{full_repo_id}")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Failed to publish to Hugging Face: {e}")

if __name__ == "__main__":
    publish()
