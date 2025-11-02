import numpy as np
import random
import time
from itertools import combinations
from collections import defaultdict

from matplotlib.style.core import available

"""
模型说明
作品数（treatments）m=3000 
评审专家数（blocks） n=125
每份作品由 a=5 位专家评审
每位专家平均需要评审的作品数b=15000/125=120
s_ij表示专家i和j评审作品相同的数量
目标为最小化全部sij的标准差  1≤i<j≤n
约束为要满足a和b的数量要求
xij=1表示第j位专家评审第i个作品
"""
start_time = time.time()
class GA_BIP():
    def __init__(self):
        self.m=3000 #作品数
        self.n=125#评委数
        self.a=5#每个作品评委人数
        self.b=120#每个评委评价作品数
        self.num_pop=50#种群数
        self.max_gen=100#最大迭代次数
        self.pc=0.9 #交叉概率
        self.pm=0.01#变异概率
        self.pop=self.init_pop()#种群初始化，创建一个可行解组成的种群
        self.fitness=self.calc_fitness()#计算种群适应度
    #初始化种群，得到的是一个3000行，5列的二维数组。方法不是传统的直接随机生成，而是多重池分配 + 局部修复
    def init_pop(self):
        pop=[]
        lst=[1]*self.a+[0]*(self.n-self.a)
        for i in range(self.num_pop):
            solution = np.zeros((self.m, self.n), dtype=int)
            for i in range(self.m):
                solution[i]=random.sample(lst, self.n)
            solution=self.repair(solution)
            pop.append(solution)
        return pop
    #修复函数，保证解是可行解,在破坏解的可行性时只破坏评委的作品数量限制，保证每个作品的评委数量始终可行，因为评委数比作品数要少，这样修复起来更快。
    def repair(self,solution):
        #用默认字典记录每个评委评了多少个作品，分为多评和少评
        excess=defaultdict(int)
        deficit=defaultdict(int)
        for j in range(self.n):
            review_count=np.sum(solution[:, j])#计算评委j评了多少个作品，有可能多也可能少
            if review_count>self.b:
                excess[j]=review_count
            elif review_count<self.b:
                deficit[j]=review_count
        #通过上面对每个评委的遍历已经得到了两个记录异常评委及其评审作品数量的字典。
        while excess:#当字典为空时说明每个评委都没有多评，那此时一定也没有评委是少评的，所以循环停止。
            a_key=min(excess,key=excess.get)#找到多评数量最少和少评数量最少的两个评委，这样可以让多评和少评的评委数尽快减少，如果有多个该函数也只会返回第一个对应的键，
            b_key=max(deficit,key=deficit.get)
            #分别找到两个评委都评了哪些作品，index_a的数量肯定是大于index_b，所以一定会至少有一个a有而b没有的作品
            index_a=np.where(solution[:, a_key] > 0)[0]
            index_b=np.where(solution[:, b_key] > 0)[0]
            available=index_a[~np.isin(index_a, index_b)]
            selected_index=available[0]#从所有可选索引中选第一个即可
            solution[selected_index][a_key]=0
            solution[selected_index][b_key]=1
            excess[a_key]-=1
            deficit[b_key]+=1
            if excess[a_key]==self.b:
                del excess[a_key]
            if deficit[b_key]==self.b:
                del deficit[b_key]
        return solution
    #计算可行解的目标函数值即标准差(standard deviation)
    def calc_std(self,solution):
        overlaps = []
        for i, j in combinations(range(self.n), 2):#i一定会小于j，即我们计算两两评委的交叉作品数量时只计算上半区的。
            overlap = np.sum(solution[:, i] * solution[:, j])  #如果两者都评价了某个作品，那么乘积肯定是1。
            overlaps.append(overlap)
        return np.std(overlaps)
    #计算种群适应度
    def calc_fitness(self):
        func_values = []
        for chromosome in self.pop:
            std = self.calc_std(chromosome)
            func_values.append(std)
        return 1/np.array(func_values)
    #选择算子，轮盘赌
    def selection(self):
        sum_fitness=sum(self.fitness)
        p=[i/sum_fitness for i in self.fitness]
        rand = np.random.rand()
        for i, sub in enumerate(p):
            if rand >= 0:
                rand -= sub
                if rand < 0:
                    index = i
        return self.pop[index].copy()#NumPy 中 copy() 就是深拷贝
    def crossover(self,p1,p2):
        c1,c2=p1.copy(),p2.copy()
        #随机选一半作品进行交叉，这些作品的评委发生改变。
        idx = random.sample(range(self.m), self.m// 2)
        c1[idx, :], c2[idx, :] = p2[idx, :], p1[idx, :]
        c1=self.repair(c1)
        c2=self.repair(c2)
        return c1, c2
    def mutate(self,child):
        #选取两个作品i和j，将这两个作品的两个特有的评委交换。
        i, j = random.sample(range(self.m), 2)
        i_reviewers=np.where(child[i]>0)[0]
        j_reviewers=np.where(child[j] > 0)[0]
        available_i = i_reviewers[~np.isin(i_reviewers, j_reviewers)]
        available_j=j_reviewers[~np.isin(j_reviewers, i_reviewers)]
        while len(available_i) == 0 or len(available_j) == 0:
            i, j = random.sample(range(self.m), 2)
            i_reviewers = np.where(child[i] > 0)[0]
            j_reviewers = np.where(child[j] > 0)[0]
            available_i = i_reviewers[~np.isin(i_reviewers, j_reviewers)]
            available_j = j_reviewers[~np.isin(j_reviewers, i_reviewers)]
        a=available_i[0]
        b=available_j[0]
        child[i, a], child[i, b] = 0, 1
        child[j, a], child[j, b] = 1, 0
        return child
    def main(self):
        print(f"当前为第{0}代，最小标准差为：{1/max(self.fitness):.8f}")
        for i in range(self.max_gen):
            new_pop=[]
            elite_index=np.argmax(self.fitness)
            new_pop.append(self.pop[elite_index])
            while len(new_pop)<self.num_pop:
                p1=self.selection()#得到的p1和p2是完全独立于self.pop的，因为其是深拷贝得到的副本。
                p2=self.selection()
                if random.random()<self.pc:
                    c1,c2=self.crossover(p1,p2)
                else:
                    c1,c2=p1,p2
                if random.random()<self.pm:
                    c1=self.mutate(c1)
                if random.random()<self.pm:
                    c2=self.mutate(c2)
                c1_std=self.calc_std(c1)
                c2_std=self.calc_std(c2)
                if c1_std<c2_std:
                    new_pop.append(c1)
                else:
                    new_pop.append(c2)
            self.pop=new_pop
            self.fitness=self.calc_fitness()
            print(f"当前为第{i+1}代，最小标准差为：{1/max(self.fitness):.8f}")

model=GA_BIP()
model.main()
end_time = time.time()
print("代码运行时间：", end_time - start_time, "秒")

