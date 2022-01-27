import numpy as np

#计算λ（学习率）
def compute_lr(x1,x2):
    a = (16*x1*x1 + 4*x2*x2) / (64*x1*x1 + 8*x2*x2)
    return a
#计算梯度
def grad(x1,x2):
    return 4*x1,2*x2

if __name__ == "__main__":
    x = [1,1]
    x = np.array(x)
    for i in range(10):
        a,b = grad(x[0], x[1])
        gradient = np.array([a,b])
        
        lr = compute_lr(x[0],x[1])
        x = x - lr*gradient
        print("x%d = "%(i+1),x)
        print("第%d梯度为: "%(i+1),gradient)
        