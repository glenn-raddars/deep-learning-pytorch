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

3. ## 初始化tensor的一些方法：

   1. rand:

      使用方法torch.rand(dim,dim)，生成制定维度的张量，并用0~1之间的随机数填充

   2. rand_like:

      使用方法：torch.rand_like(a)，输入一个张量，然后生成跟这个张量维度相同的随机张量

   3. randint：

      使用方法：torch.randint(a,b,c)，几个参数分别是随机数起始数，随机数末尾数，张量维度，如torch.randint(1,10,[3,3])生成维度为3*3的张量，然后用1~9的数字填充，不会超过10

   4. randint_like:

      跟rand_like相同
      
   5. full
   
      使用方法：torch.full(size, fill_value)，前面是张量的形状，后面是需要填充的值，如torch.full([2,3], fill_value = 7)，输出为tensor([[7, 7, 7], [7, 7, 7]])
   
   6. linspace
   
      使用方法：torch.linspace(),如torch.linspace(1, 10,steps=10)，生成一个一维张量，从1到10的等差数列，步长为1（10等分）
   
   7. logspace
   
      使用方法：torch.logspace(),如torch.logspace(1, 10,steps=10)，生成一个一维张量，里面的数字是，以10为底数，1到10且步长为1的等差数列作为指数的张量。
   
   8. randperm
   
      使用方法：torch.randperm(n)，生成从0到n-1随机种子，可以用于张量在dim=0的维度上的交换。
   
4. ## 对Tensor进行提取操作的方法

   1. index_select

      使用方法：torch.index_select()，a.index_select( dim, index)，在哪个维度进行提取，提取的目标索引值(且索引值必须是一个tensor实例对象)，如

      ```python
      a = torch.ones([4,3,28,28])
      b = torch.tensor([1,2])
      a = a.index_select(0,b)
      ```

      此时a的size为(2,3,28,28),实际上选取的就是第零维的1,2两个元素。

   2. ...

      使用方法，a[...]，意思就是a，或者a[0,...]意思就是a[0]，...就是取后面所有的。

5. ## 对Tensor的形变操作

   1. view

      老朋友，没什么好说的，改变tensor的形状，a.view()

   2. unsqueeze

      使用方法：a.unsqueeze(n)，在索引值所在地新增一个维度，如

      ```python
      a = torch.rand([4,1,28,28])
      a.unsqueeze(1)
      ```

      会使a变成[4,1,1,28,28]，在1处新增加了一个维度1

   3. squeeze

      使用方法：a.squeeze(n),在索引值所在地去除一个维度，如

      ```python
      a = torch.rand([1,32,1,1])
      a.squeeze()
      ```

      a就会直接变成一个32

      ```python
      a.squeeze(0)
      ```

      a变成了[32,1,1]，（当索要挤压的维度不唯一时，返回原来的张量）

   4. expand

      扩张，使用方法：a.expand(...)，只有原来维度上只有1的维度才能扩展，本质就是复制粘贴

      ```python
      a = torch.rnad([1,2,2])
      a = a.expand(2,-1,-1)
      ```

      此时a的size为[2,2,2]，多加的一个维度就是把原来的数据复制粘贴一份
      
   5. repeat

     使用方法：a.repeat(...),跟expand有些许的不同，这个的本意是在一个维度上把原来的重复多少次
     
     ```python
     a = torch.rand([1,2,2])
     a = a.repeat(2,2,2)
     ```
     
      最后a的size为[2,4,4]
     
   6. t（转置）
   
      使用方法：a.t()，居镇专职没什么好说的，只能用于2Dtensor
   
   7. transpose
   
      使用方法：a.transpose(...)，挑两个维度进行转置
   
      ```python
      a = torch.tensor([[1,2,3],[3,4,5]])
      a = a.unsqueeze(0)
      a = a.transpose(0,2) 
      ```
   
      此时a的size为[3,2,1]，a已经变成了
   
      ```
      tensor([[[1],
               [3]],
      
              [[2],
               [4]],
      
              [[3],
               [5]]])
      ```
   
   8. permute
   
      使用方法：a.permute(...)，将原来的维度转换到对应位置上去（太抽象了，直接看代码）
   
      ```python
      a = torch.tensor([[1,2,3],[3,4,5]])
      a = a.unsqueeze(0)
      a = a.transpose(0,1)
      a = a.transpose(1,2)
      print(a.size())
      a = a.permute(2,0,1)
      print(a.shape)
      ```
   
      输出为
   
      ```
      torch.Size([2, 3, 1])
      torch.Size([1, 2, 3])
      ```
   
      就是将原来2维度对应的数值1，放到了新张量中第0维的位置，后面同理























