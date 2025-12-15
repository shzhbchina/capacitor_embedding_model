import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
print('if set break here, paste below part. It will cause window busy and no plot will show')
plt.figure()
plt.plot([1,2,3], [4,5,6])
plt.show()  # 没有这一步本地不会弹出图像内容