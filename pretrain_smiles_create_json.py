import os
import json
from random import shuffle

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
from tqdm import tqdm


# 标准化处理
def canonicalize(s):
    return Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True,
                            isomericSmiles=False)


# 读入要处理的数据
df = pd.read_csv('data/12999.csv')
canonical_smiles = df.smiles.apply(canonicalize)  # SMILES字符串进行标准化处理，处理后的结果存储在名为 canonical_smiles的对象中
index_data = df['index'].tolist()
smiles_data = canonical_smiles.values.tolist()


def generate_json_for_images(images_folder_path, smiles_data, index_data):
    data = []

    for filename, smiles, index in tqdm(zip(os.listdir(images_folder_path), smiles_data, index_data)):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # 只处理特定图片格式的文件
            image_path = os.path.join(images_folder_path, str(index)).replace('\\', '/')  # 将反斜杠替换为正斜杠
            image_path = os.path.join(image_path, str('.png')).replace('\\', '')
            image_path = os.path.join(str('pretrain_smiles'), image_path).replace('\\', '/')
            caption = smiles  # 通过输入获取图片描述，也可根据需要修改为其他方式获取描述信息
            image_data = {
                "image": image_path,
                "smiles": caption,

            }
            data.append(image_data)
    # 随机打乱数据顺序
    shuffle(data)

    # 分割数据集
    train_data = data[:int(0.3 * len(data))]  # 70% 的数据用于训练集

    val_data = data[int(0.7 * len(data)):int(0.85 * len(data))]  # 15% 的数据用于验证集
    test_data = data[int(0.85 * len(data)):]  # 15% 的数据用于测试集

    # 写入训练集数据到 train.json
    with open('pretrain_smiles_train.json', 'w') as train_file:
        json.dump(train_data, train_file, indent=4)

    # 写入验证集数据到 val.json
    test_data_without_id = [{"image": entry["image"], "smiles": entry["smiles"]} for entry in val_data]
    with open('pretrain_smiles_val.json', 'w') as test_file:
        json.dump(test_data_without_id, test_file, indent=4)

    # 写入测试集数据到 test.json，不包含 image_id
    test_data_without_id = [{"image": entry["image"], "smiles": entry["smiles"]} for entry in test_data]
    with open('pretrain_smiles_test.json', 'w') as test_file:
        json.dump(test_data_without_id, test_file, indent=4)


images_folder_path = "data/images"
# 调用函数并传入图片文件夹的路径
generate_json_for_images(images_folder_path, smiles_data, index_data)
