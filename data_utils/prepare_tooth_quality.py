import struct
import os
from stl import mesh
import trimesh

def convert_to_txt(source_file, target_file):
    tooth = mesh.Mesh.from_file(source_file)
    # 这里我们把面片当成点云进行处理即可

    # 面片的法向量
    normals = tooth.normals

    assert len(normals) > 0

    # 面片的中点
    centroids = tooth.centroids

    # 把点云信息写入到txt
    with open(target_file, 'w', encoding='ascii') as f:
        for centroid, normal in zip(centroids, normals):
            f.write(f'{centroid[0]},{centroid[1]},{centroid[2]},{normal[0]},{normal[1]},{normal[2]}\n')
            f.flush()


def downsample(source_path, target_path):
    files = os.listdir(source_path)

    for file in files:
        print(file)
        # 加载STL文件
        mesh = trimesh.load_mesh(os.path.join(source_path, file))

        # 使用trimesh进行降采样
        downsampled_mesh = mesh.simplify_quadratic_decimation(10000)

        # 将降采样后的模型保存为新的STL文件
        downsampled_mesh.export(os.path.join(target_path, file))


def batch(normal_src, normal_tgt, abnormal_src, abnormal_tgt):
    normal_files = os.listdir(normal_src)
    train_info = ""

    for file in normal_files:
        # 把文件名字换成我们希望的名字
        bare_name = file.split('.')[0]
        code = bare_name[2:]

        tgt_file = 'normal_' + code + '.txt'
        train_info += 'normal_' + code + '\n'

        convert_to_txt(os.path.join(normal_src, file), os.path.join(normal_tgt, tgt_file))

    abnormal_files = os.listdir(abnormal_src)
    for file in abnormal_files:
        # 文件名替换
        bare_name = file.split('.')[0]
        code = bare_name[2:]
        tgt_file = 'abnormal_' + code + '.txt'
        train_info += 'abnormal_' + code + '\n'

        convert_to_txt(os.path.join(abnormal_src, file), os.path.join(abnormal_tgt, tgt_file))

    # 写入训练文件
    with open('D:/Dataset/Tooth_quality/tooth_quality_train.txt', 'w') as f:
        f.write(train_info)


if __name__ == '__main__':
    normal_src = 'D:/Dataset/Tooth_quality/normal_downsample'
    normal_tgt = 'D:/Dataset/Tooth_quality/normal'
    abnormal_src = 'D:/Dataset/Tooth_quality/abnormal_downsample'
    abnormal_tgt = 'D:/Dataset/Tooth_quality/abnormal'
    batch(normal_src, normal_tgt, abnormal_src, abnormal_tgt)
    # downsample("D:\\Dataset\\Tooth_quality\\normal_origin", "D:\\Dataset\\Tooth_quality\\normal_downsample")
