# 关于Tensor的相关方法

1. ## 判断tensor的类型：

   1. type：

      type可以判断tensor的类型，使用方法为a.type()，或者type(a)(但此方法不常用)

   2. isinstance

      isinstance可以用来判断张量的数据类型，使用方法为isinstance(data,type)它会返回布尔类型的值

2. ##  判断tensor的形状大小：

   1. shape：

      使用方法，a.shap，返回一个torch.tensor([])，里面会显示这是一个几行几列的张量，不显示默认为0

   2. size():

      使用方法：a.size()，返回与shape相同

   3. dim():

      使用方法：a.dim()，直接返回张量的维度，如：矩阵的维度是2

   4. numel()：

      使用方法：a.numel()，直接返回一个张量所占内存的大小，就是所有维度相乘

3. ##初始化tensor的一些方法：

   1. rand:

      使用方法torch.rand(dim,dim)，生成制定维度的张量，并用0~1之间的随机数填充

   2. rand_like:

      使用方法：torch.rand_like(a)，输入一个张量，然后生成跟这个张量维度相同的随机张量

   3. randint：

      使用方法：torch.randint(a,b,c)，几个参数分别是随机数起始数，随机数末尾数，张量维度，如torch.randint(1,10,[3,3])生成维度为3*3的张量，然后用1~9的数字填充，不会超过10

   4. randint_like:

      跟rand_like相同























