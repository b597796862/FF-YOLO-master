#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
from nets.yolo import yolo_body
from utils.utils import net_flops
import numpy as np
import time
if __name__ == "__main__":
    input_shape = [640, 640]
    num_classes = 1
    phi         = 'tiny'

    model = yolo_body([input_shape[0], input_shape[1], 3], num_classes, phi)
    #--------------------------------------------#
    #   查看网络结构网络结构
    #--------------------------------------------#
    model.summary()
    #--------------------------------------------#
    #   计算网络的FLOPS
    #--------------------------------------------#
    net_flops(model, table=False)
    
    #--------------------------------------------#
    #   获得网络每个层的名称与序号
    #--------------------------------------------#
    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name)
    # 计算推理时间
    fimage = np.ndarray(shape=[1, input_shape[0], input_shape[1], 3])
    sum=0
    for i in range(20):
        start_time = time.time()
        model(fimage)
        end_time = time.time()  # 记录程序结束运行时间
        sum+=(end_time - start_time)
    print(sum / 20)
    print("fps=",1/(sum/20))