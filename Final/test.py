import open3d as o3d

input = "/mnt/d/1112_CV/Final/ITRI_dataset/seq1/localization_timestamp.txt"
f = open(input,"r")
lines = f.readlines()
print(len(lines))
tmp = []
error = []
for line in lines:
    if line not in tmp:
        tmp.append(line)
    else:
        error.append(line)
print(len(tmp))
print(len(error), error)


"""
with open("test.txt",'w') as f:
    for i in range(3,len(lines)):
        xyz = lines[i].split(" ")[1:4]
        f.write(xyz[0]+" "+xyz[1]+" "+xyz[2]+"\n")

pcd = o3d.io.read_point_cloud('test.txt', format='xyz')
o3d.visualization.draw_geometries([pcd])
"""