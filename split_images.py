import os
from PIL import Image

def split_image_with_overlap(input_image_path, output_dir, patch_size=256, overlap=128):
    """
    一枚の画像をオーバーラップさせながら複数のパッチに分割する関数
    （この関数の内容は前回と同じです）
    """
    try:
        image = Image.open(input_image_path)
        img_width, img_height = image.size
        
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(input_image_path))[0]
        step = patch_size - overlap
        patch_count = 0
        
        for y in range(0, img_height, step):
            if y + patch_size > img_height:
                continue
            for x in range(0, img_width, step):
                if x + patch_size > img_width:
                    continue
                
                box = (x, y, x + patch_size, y + patch_size)
                patch = image.crop(box)
                
                output_filename = f"{base_name}_{patch_count:04d}.png"
                output_path = os.path.join(output_dir, output_filename)
                patch.save(output_path)
                
                patch_count += 1
        
        print(f"完了: {input_image_path} を {patch_count} 個のパッチに分割しました。")

    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {input_image_path}")
    except Exception as e:
        print(f"エラーが発生しました ({input_image_path}): {e}")

def process_all_images_in_folder(input_dir, output_dir):
    """
    指定されたフォルダ内のすべての画像ファイルを処理する関数
    """
    # 処理対象とする画像の拡張子
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif')
    
    # 入力フォルダ内のすべてのファイルとディレクトリを取得
    for filename in os.listdir(input_dir):
        # ファイルの拡張子がサポートされている形式かチェック（小文字に変換して比較）
        if filename.lower().endswith(supported_formats):
            # 入力画像のフルパスを作成
            input_image_path = os.path.join(input_dir, filename)
            # パッチ分割関数を呼び出す
            split_image_with_overlap(input_image_path, output_dir)

# --- ここから設定 ---

# 1. 画像が入っているフォルダのパスを指定
input_folder = 'livecell/A172_image_fomerhalf' 

# 2. 分割したパッチを保存するフォルダ名を指定
output_folder = 'livecell/A172_fomerhalf/image/0000'

# --- 実行 ---
if __name__ == '__main__':
    # フォルダ内の全画像を処理する関数を実行
    process_all_images_in_folder(input_folder, output_folder)