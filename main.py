import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import norm

class ElectrochemicalBatteryModel:
    """电池电化学反应动力学模型"""

    def __init__(self, capacity, resistance, voltage, temperature=25, max_soc=1.0, min_soc=0.0, nominal_voltage=3.7):
        self.capacity = capacity  # 电池容量 (Ah)
        self.resistance = resistance  # 内部电阻 (ohm)
        self.voltage = voltage  # 电池电压 (V)
        self.nominal_voltage = nominal_voltage  # 电池额定电压 (V)
        self.temperature = temperature  # 电池温度 (°C)
        self.state_of_charge = 1.0  # 电池初始SOC

    def calculate_voltage(self, soc):
        """
        计算电池电压，考虑SOC和T~
        :param soc: 电池的充电状态 (SOC)
        :return: 电池电压 (V)
        """
        voltage_drop = self.resistance * (self.voltage - self.nominal_voltage)  # 电池内部电阻的电压降
        voltage = self.nominal_voltage * soc - voltage_drop
        return voltage

    def simulate(self, current, time_steps=10):
        """
        仿真电池电压，模拟不同SOC下的电池行为。
        :param current: 电池电流 (A)
        :param time_steps: 仿真时长（单位：秒）
        :return: 电池电压随时间的变化
        """
        voltage_values = []
        for t in range(time_steps):
            # 更新电池SOC
            self.state_of_charge += current * 1  # 简单更新公式
            self.state_of_charge = np.clip(self.state_of_charge, 0, 1)  # 限制SOC在0-1之间
            voltage = self.calculate_voltage(self.state_of_charge)
            voltage_values.append(voltage)
        return np.array(voltage_values)


def least_squares_optimization(model, time_steps, experimental_data, initial_params):
    """
    使用最小二乘法优化电池模型参数，使得模型预测与实验数据之间的误差最小化。
    :param model: 电池模型类
    :param time_steps: 仿真时间步数
    :param experimental_data: 实验数据（实际电压数据）
    :param initial_params: 初始参数猜测
    :return: 优化后的参数
    """

    def objective_function(params):
        """
        目标函数，计算模型预测电压与实验电压的差异（最小二乘误差）。
        :param params: 模型参数 (如电池容量、内阻等)
        :return: 最小化的误差值
        """
        model.capacity, model.resistance = params[0], params[1]  # 更新模型参数
        predicted_voltage = model.simulate(current=-5, time_steps=time_steps)  # 仿真电池电压
        error = np.sum((predicted_voltage - experimental_data) ** 2)  # 计算误差（最小二乘法）
        return error

    # 使用最小二乘法进行参数优化
    result = minimize(objective_function, initial_params, method='Nelder-Mead')
    optimized_params = result.x
    return optimized_params


def bayesian_optimization(model, time_steps, experimental_data, prior_mean, prior_std, num_iterations=100):
    """
    使用贝叶斯推断优化电池模型参数，根据个人或实验室需要，可以结合先验信息和实验数据进行参数更新。
    :param model: 电池模型类
    :param time_steps: 仿真时间步数
    :param experimental_data: 实验数据（实际电压数据）
    :param prior_mean: 参数的先验均值
    :param prior_std: 参数的先验标准差
    :param num_iterations: 迭代次数
    :return: 后验分布的均值和标准差
    """
    posterior_mean = np.array(prior_mean)  # 初始后验均值为先验均值
    posterior_std = np.array(prior_std)  # 初始后验标准差为先验标准差

    # 贝叶斯更新过程
    for _ in range(num_iterations):
        # 计算似然函数：基于当前参数仿真，计算与实验数据的差异
        def likelihood(params):
            model.capacity, model.resistance = params[0], params[1]  # 更新模型参数
            predicted_voltage = model.simulate(current=-5, time_steps=time_steps)
            likelihood_value = -np.sum((predicted_voltage - experimental_data) ** 2)  # 负的最小二乘误差
            return likelihood_value

        # 采样：基于当前后验分布采样
        sampled_params = np.random.normal(posterior_mean, posterior_std)
        likelihood_value = likelihood(sampled_params)

        # 更新后验分布
        posterior_mean += 0.1 * (sampled_params - posterior_mean)  # 更新均值
        posterior_std += 0.1 * np.abs(sampled_params - posterior_mean)  # 更新标准差

    return posterior_mean, posterior_std


# 主函数
if __name__ == "__main__":
    # 定义电池模型
    battery_model = ElectrochemicalBatteryModel(capacity=50, resistance=0.1, voltage=3.7)

    # 假设的实验数据（实际使用时应从实验中获得）
    time_steps = 10
    experimental_data = np.array([3.5, 3.45, 3.42, 3.4, 3.35, 3.3, 3.25, 3.2, 3.1, 3.05])

    # 初始参数猜测
    initial_params = [50, 0.1]  # 初始猜测的电池容量和内阻

    # 1. 最小二乘法优化
    optimized_params_ls = least_squares_optimization(battery_model, time_steps, experimental_data, initial_params)
    print(f"最小二乘法优化结果：电池容量 = {optimized_params_ls[0]}, 内阻 = {optimized_params_ls[1]}")

    # 2. 贝叶斯推断优化
    prior_mean = [50, 0.1]  # 先验均值
    prior_std = [5, 0.01]  # 先验标准差
    posterior_mean, posterior_std = bayesian_optimization(battery_model, time_steps, experimental_data, prior_mean,
                                                          prior_std)
    print(f"贝叶斯推断优化结果：电池容量 = {posterior_mean[0]}, 内阻 = {posterior_mean[1]}")

    # 绘制结果对比
    plt.plot(experimental_data, label='实验数据')
    plt.plot(battery_model.simulate(current=-5, time_steps=time_steps), label='最小二乘法优化模型')
    plt.legend()
    plt.title("实验数据与优化结果对比")
    plt.xlabel("时间步长 (秒)")
    plt.ylabel("电池电压 (V)")
    plt.show()
