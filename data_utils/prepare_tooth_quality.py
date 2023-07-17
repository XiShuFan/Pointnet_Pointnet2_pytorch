"""
把二进制文件转换成点云文件，ply格式
"""


"""
float startBox[3], endBox[3], origin[3];
std::vector<float> solidSDF;	
float	 dx;//每个体素大小
int volNum;						//体素数量
int ni, nj, nk;					//三个方向上的分辨率


//读入volName
volFullName = fullName + ".vol";
FILE *volFile = fopen(volFullName.c_str(), "rb");
//读入体素信息
fread(&ni, sizeof(int), 1, volFile); 
fread(&nj, sizeof(int), 1, volFile); 
fread(&nk, sizeof(int), 1, volFile);		
//读入关键信息
float point[3]; 
fread(point, sizeof(float), 3, volFile); 
origin = Vertex3(point[0], point[1], point[2]);
//读入极小值点和极大值点
fread(startBox, sizeof(float), 3, volFile); 
fread(endBox, sizeof(float), 3, volFile);
//读取单位补偿
fread(&dx, sizeof(float), 1, volFile);
//读入体素数据
volNum = ni*nj*nk; 
solidSDF.resize(volNum); 
fread(&solidSDF[0], sizeof(float), volNum, volFile);

书帆，你写个小程序，把这些体素读进来。体素值等于0的，是表面点

"""


import struct
import os


def convert_to_ply(fullName: str, target: str):
    # 读入 volumn
    volFile = open(fullName, "rb")

    # 读入体素信息
    ni = struct.unpack('i', volFile.read(4))[0]
    nj = struct.unpack('i', volFile.read(4))[0]
    nk = struct.unpack('i', volFile.read(4))[0]

    # 读入关键信息
    point = struct.unpack('fff', volFile.read(12))
    origin = [point[0], point[1], point[2]]

    # 读入极小值点和极大值点
    startBox = list(struct.unpack('fff', volFile.read(12)))
    endBox = list(struct.unpack('fff', volFile.read(12)))

    # 读取单位补偿
    dx = struct.unpack('f', volFile.read(4))[0]

    # 读入体素数据
    volNum = ni * nj * nk
    solidSDF = list(struct.unpack('f'*volNum, volFile.read(4*volNum)))

    # 遍历所有的坐标点，输出到ply文件中
    vertex_info = ""
    with open(target, 'w', encoding='ascii') as f:
        count = 0
        for x in range(ni):
            for y in range(nj):
                for z in range(nk):
                    # 只需要判断表面点
                    if abs(solidSDF[x * nj * nk + y * nk + z]) < 1e-2:
                        coor_x = startBox[0] - x * dx
                        coor_y = startBox[1] - y * dx
                        coor_z = startBox[2] - z * dx
                        vertex_info += f'{coor_x},{coor_y},{coor_z}\n'
                        count += 1
        # 直接按照txt文本写入点坐标
        f.write(vertex_info)

    volFile.close()


def batch(normal_src, normal_tgt, abnormal_src, abnormal_tgt):
    normal_files = os.listdir(normal_src)
    train_info = ""

    for file in normal_files:
        # 把文件名字换成我们希望的名字
        bare_name = file.split('.')[0]
        code = bare_name[2:]

        tgt_file = 'normal_' + code + '.txt'
        train_info += 'normal_' + code + '\n'

        convert_to_ply(os.path.join(normal_src, file), os.path.join(normal_tgt, tgt_file))

    abnormal_files = os.listdir(abnormal_src)
    for file in abnormal_files:
        # 文件名替换
        bare_name = file.split('.')[0]
        code = bare_name[2:]
        tgt_file = 'abnormal_' + code + '.txt'
        train_info += 'abnormal_' + code + '\n'

        convert_to_ply(os.path.join(abnormal_src, file), os.path.join(abnormal_tgt, tgt_file))

    # 写入训练文件
    with open('/media/why/77B8B456EE73FE06/users/xsf_ubuntu/Dataset/Tooth_quality/tooth_quality_train.txt', 'w') as f:
        f.write(train_info)


if __name__ == '__main__':
    # convert_to_ply("D:\\Dataset\\Tooth_quality\\abnormal\\TN050.vol", "D:\\Dataset\\Tooth_quality\\abnormal_ply\\TN050.ply")
    normal_src = '/media/why/77B8B456EE73FE06/users/xsf_ubuntu/Dataset/Tooth_quality/positive_origin'
    normal_tgt = '/media/why/77B8B456EE73FE06/users/xsf_ubuntu/Dataset/Tooth_quality/normal'
    abnormal_src = '/media/why/77B8B456EE73FE06/users/xsf_ubuntu/Dataset/Tooth_quality/negative_origin'
    abnormal_tgt = '/media/why/77B8B456EE73FE06/users/xsf_ubuntu/Dataset/Tooth_quality/abnormal'
    batch(normal_src, normal_tgt, abnormal_src, abnormal_tgt)
