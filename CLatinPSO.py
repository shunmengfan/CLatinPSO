# coding=utf-8
import heapq
import random
from functools import partial
import numpy as np


def _obj_wrapper(func, args, kwargs, x):
    return func(x, *args, **kwargs)  # 执行func函数


def _is_feasible_wrapper(func, x):
    return np.all(func(x) >= 0)


def _cons_none_wrapper(x):
    return np.array([0])


def _cons_ieqcons_wrapper(ieqcons, args, kwargs, x):
    return np.array([y(x, *args, **kwargs) for y in ieqcons])


def _cons_f_ieqcons_wrapper(f_ieqcons, args, kwargs, x):
    return np.array(f_ieqcons(x, *args, **kwargs))


def clatinpso(func, lb, ub, i, args=(), kwargs={},  # func是一个对象
        swarmsize=40, maxiter=100,
        debug=True, particle_output=False, processes=6, ieqcons=[], f_ieqcons=None, minfunc=0,):
    assert len(lb) == len(ub), 'Lower- and upper-bounds must be the same length'  # 参数上下限长度相等否则报错
    assert hasattr(func, '__call__'), 'Invalid function handle'  # 若func对象没有__call__方法则报错
    lb = np.array(lb) # 创建矩阵
    ub = np.array(ub)
    num = i
    assert np.all(ub > lb), 'All upper-bound values must be greater than lower-bound values'  # 上限大于下限
    # mp_pool = multiprocessing.Pool(6)  # 开6个进程？
    interval = np.abs(ub - lb)  # 上下限差距
    vhigh = 0.2 * interval  # 限定速度范围
    vlow = -vhigh

    # 初始化目标函数
    obj = partial(_obj_wrapper, func, args, kwargs)  # 执行_obj_wrapper函数，也就是func函数，后面三个是固定的参数
    D = len(lb)  # 粒子的维度15

    # 学习率 Pc = np.ones([D, 1]) * (0.05 + (0.45 * (np.exp(j) - np.exp(j[0]))/(np.exp(j[swarmsize - 1]) - np.exp(j[0]))))
    Pc = np.zeros(swarmsize)  # 100个粒子的学习率
    for i in range(swarmsize):
        Pc[i] = 0.0 + 0.5 * (np.exp(5 * i / swarmsize - 1) - 1) / (np.exp(5) - 1)

    ite = np.linspace(1, maxiter, maxiter)  # List[1,2,3,...,100]
    Weight = 0.9 - ite * 0.7 / maxiter  # 权重w，随着迭代次数线性减小
    c0 = 0.5
    c1 = 1.5
    c2 = 1.5

    # 检查约束函数 #########################################
    if f_ieqcons is None:
        if not len(ieqcons):
            if debug:
                print('No constraints given.') # 没有约束函数
            cons = _cons_none_wrapper
        else:
            if debug:
                print('Converting ieqcons to a single constraint function') # ieqcons转换为单个约束函数
            cons = partial(_cons_ieqcons_wrapper, ieqcons, args, kwargs)
    else:
        if debug:
            print('Single constraint function given in f_ieqcons') # f_ieqcons给出的单个约束函数
        cons = partial(_cons_f_ieqcons_wrapper, f_ieqcons, args, kwargs)
    is_feasible = partial(_is_feasible_wrapper, cons)
    # 初始化多进程模块（若需要的话）
    # if processes > 1:
    #     import multiprocessing
    #     mp_pool = multiprocessing.Pool(processes)
    # 初始化粒子群 ############################################

    x = np.random.rand(swarmsize, D)  # 改成Latin采样
    fx = np.zeros(swarmsize)  # 粒子适应度初始化数组
    fs = np.zeros(swarmsize, dtype=bool)  # 每个粒子的可行性
    p = np.zeros_like(x)  # pbest位置初始化
    fp = np.ones(swarmsize) * np.inf  # pbest适应度，初始化为1000个正无穷
    g = []  # gbest位置
    fg = np.inf  # gbest适应度，初始化为正无穷

    # 初始化粒子位置
    dis = ub - lb # 上限 - 下限
    dis = dis / 2.0  # 区间长度的一半

    """dis = [1 1 1 1 
        1 1 1 1
        1 1 1 1
        96 96 96]"""
    x_initvalue = np.zeros_like(x)  # 100 * 15
    for i_init in range(int(swarmsize / 2)):
        for j_init in range(D):
            randvalue = random.sample(range(2), 2)
            for k_init, value_init in enumerate(randvalue):
                x_initvalue[i_init * 2 + k_init][j_init] = value_init

    for l_init in range(swarmsize):  # 每个粒子用拉丁采样的方式随机选取15维度的参数（有范围）
        for m_init in range(D):
            x[l_init][m_init] = np.random.uniform(lb[m_init] + dis[m_init] * x_initvalue[l_init][m_init],
                                                  lb[m_init] + dis[m_init] * (x_initvalue[l_init][m_init] + 1))
    # x = lb + x*(ub - lb)
    # Calculate objective and constraints for each particle
    if processes > 1:
        print()
        # fx = np.array(mp_pool.map(obj, x))  # 开多个进程求适应度
        # fs = np.array(mp_pool.map(is_feasible, x))  # 求可行性
    else:
        for i in range(swarmsize):
            fx[i] = obj(x[i, :])  # 直接求适应度
            fs[i] = is_feasible(x[i, :])
        # 存储pbest (约束条件满足条件下)
        i_update = np.logical_and((fx < fp), fs)  # 可行且当前比最佳还小情况下，置true
        p[i_update, :] = x[i_update, :].copy()  # 更新pbest位置
        fp[i_update] = fx[i_update]  # 更新pbest适应度
        # 更新gbest
        i_min = np.argmin(fp)
        if fp[i_min] < fg:
            fg = fp[i_min]  # 更新gbest适应度
            g = p[i_min, :].copy()  # 更新gbest位置
        else:
            # 初始化给定临时gbest
            g = x[0, :].copy()
        # 初始化粒子速度
    v = vlow + np.random.rand(swarmsize, D) * (vhigh - vlow)  # 随机初始化粒子速度 vlow = -vhigh vhigh = np.abs(ub - lb) 1000*15

    # 计算每个粒子的目标和约束条件，直到满足终止标准 ##################################
    update_flag = np.zeros([swarmsize, 1])  # 停滞间隙，1000个元素的List初始化为1
    it = 0  # 迭代计数器初始化
    vel = np.zeros_like(v)  # 1000*15的全0二维数组

    x_bad = int(0.05 * swarmsize)  # 种群的1/20
    x_nose = np.random.rand(x_bad, D)  # 初始化1/10的小种群
    f_temp_x = np.zeros(swarmsize)  # 初始化1000个0

    while it <= maxiter - 1:
        for i in range(swarmsize):  # 100个粒子迭代
            if update_flag[i] >= 7:  # 停滞间隙为7 全部维度一起的
                vel[i, :] = Weight[it] * v[i, :] + (c1 * np.random.rand(1, D) * (p[i, :] - x[i, :])) + (
                        c2 * np.random.rand(1, D) * (g - x[i, :]))  # 与PSO接近，但是多了一个w，也是学自己
                update_flag[i] = 0
            else:
                for j in range(D):  # 每个维度分别
                    if np.random.rand() > Pc[i]:
                        vel[i, j] = Weight[it] * v[i, j] + (
                                c0 * np.random.rand() * (p[i, j] - x[i, j]))  # 以自己pbest为学习样例
                    else:
                        friend1 = np.ceil((swarmsize - 1) * np.random.rand())  # 以其它两个粒子pbest为学习样例
                        friend2 = np.ceil((swarmsize - 1) * np.random.rand())
                        friend1 = friend1.astype(int)
                        friend2 = friend2.astype(int)
                        if fp[friend1] < fp[friend2]:
                            vel[i, j] = Weight[it] * v[i, j] + (c0 * np.random.rand() * (p[friend1, j] - x[i, j]))
                        else:
                            vel[i, j] = Weight[it] * v[i, j] + (c0 * np.random.rand() * (p[friend2, j] - x[i, j]))
        # 速度和位置的越界处理，不知道具体怎么处理的，但是知道意义
        maskl = vel < vlow
        masku = vel > vhigh
        vel = vel * (~np.logical_or(maskl, masku)) + vlow * maskl + vhigh * masku  # 修改超出上下界值的速度vel
        pos = x[:, :] + vel  # 更新位置

        x[:, :] = pos
        v[:, :] = vel

        maskl = x < lb
        masku = x > ub
        x = x * (~np.logical_or(maskl, masku)) + lb * maskl + ub * masku  # 修改超出上下界值的位置X

        # fx = np.array(mp_pool.map(obj, x))  # 根据新的位置计算适应度
        for i in range(swarmsize):  # 更新pbest
            fx[i] = obj(x[i, :])
            if fx[i] < fp[i]:
                p[i, :] = x[i, :]
                fp[i] = fx[i]
                update_flag[i] = 0  # pbest更新过就置0
            else:
                update_flag[i] = update_flag[i] + 1  # pbest没更新过就加1，越大说明位置越好
        # 更新gbest（debug为真情况下）
        i_min = np.argmin(fp)
        if fp[i_min] < fg:
            if debug:  # 更新gbest
                print('{:} New best for swarm at iteration {:}: {:} {:}' \
                      .format(num, it, p[i_min, :], fp[i_min]))
            g = p[i_min, :].copy()
            fg = fp[i_min]

            # Latin策略
        if it % 10 == 0:  # 每迭代10次淘汰1/10较差粒子并重新拉丁取样
            x_change_initvalue = np.zeros_like(x_nose)  # （10,D)
            for i_init_in in range(int(x_bad / 2)):  # 10
                for j_init_in in range(D):
                    randvalue_in = random.sample(range(2), 2)
                    for k_init_in, value_init_in in enumerate(randvalue_in):
                        x_change_initvalue[i_init_in * 2 + k_init_in][j_init_in] = value_init_in

            for i_lost in range(swarmsize):
                f_temp_x[i_lost] = obj(x[i_lost, :])
            list_change = heapq.nlargest(x_bad, range(len(f_temp_x)), f_temp_x.take)

            j_choose_in = 0
            for i_change_in in list_change:
                update_flag[i_change_in] = 1
                for i_choose_in in range(D):
                    x[i_change_in][i_choose_in] = np.random.uniform(
                        lb[i_choose_in] + dis[i_choose_in] * x_change_initvalue[j_choose_in][i_choose_in],
                        lb[i_choose_in] + dis[i_choose_in] * (x_change_initvalue[j_choose_in][i_choose_in] + 1))
                j_choose_in = j_choose_in + 1
        # 更新目标和约束
        if processes > 1:
            # fx = np.array(mp_pool.map(obj, x))
            # fs = np.array(mp_pool.map(is_feasible, x))
            print()
        else:
            for i in range(swarmsize):
                fx[i] = obj(x[i, :])
                fs[i] = is_feasible(x[i, :])

        # 存储pbest位置 (if constraints are satisfied)
        i_update = np.logical_and((fx < fp), fs)
        p[i_update, :] = x[i_update, :].copy()
        fp[i_update] = fx[i_update]
        # 得到gbest
        i_min = np.argmin(fp)
        if fp[i_min] < fg:
            if debug:
                print('New best0 for swarm at iteration {:}: {:} {:}' \
                      .format(it, p[i_min, :], fp[i_min]))

            p_min = p[i_min, :].copy()

            if np.abs(fg - fp[i_min]) <= minfunc:
                print('Stopping search: Swarm best objective change less than {:}' \
                      .format(minfunc))
                if particle_output:
                    return p_min, fp[i_min], p, fp
                else:
                    return p_min, fp[i_min]

            else:
                g = p_min.copy()
                fg = fp[i_min]

        if debug:
            print('{:} Best after iteration {:}: {:} {:}'.format(num, it, g, fg))
        it += 1
    # 循环结束
    print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))
    if not is_feasible(g):
        print("However, the optimization couldn't find a feasible design. Sorry")
    if particle_output:
        return g, fg, p, fp
    else:
        return g, fg
