#Copyright (c) 2022-2023 Lingkai Zhu, Uppsala University, Sweden
import os

scene_directory = '/home/lingkai/lingkai/AHDRNet/GenerH5Data/Result/Training'
for file in os.listdir(scene_directory):
	print(file)
	dir = os.path.join(scene_directory, file)
	dir = dir + '\n'
	fname = 'train.txt'
	try:
		fobj = open(fname, 'a')
	except IOError:
		print('open error')
	else:
		fobj.write(dir)
		fobj.close()