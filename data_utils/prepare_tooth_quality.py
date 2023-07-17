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
                        vertex_info += f'{coor_x} {coor_y} {coor_z}\n'
                        count += 1

        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment VCGLIB generated\n")
        f.write(f"element vertex {count}\n")
        f.write("property double x\n")
        f.write("property double y\n")
        f.write("property double z\n")
        f.write("end_header\n")
        f.write(vertex_info)


    volFile.close()



if __name__ == '__main__':
    convert_to_ply("D:\\Dataset\\Tooth_quality\\abnormal\\TN050.vol", "D:\\Dataset\\Tooth_quality\\abnormal_ply\\TN050.ply")