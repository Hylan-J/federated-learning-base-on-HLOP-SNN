import h5py
import numpy as np
import os


def average_data(algorithm="", dataset="", goal="", times=10):
    test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times)

    max_accurancy = []
    for i in range(times):
        max_accurancy.append(test_acc[i].max())

    print("std for best accurancy:", np.std(max_accurancy))
    print("mean for best accurancy:", np.mean(max_accurancy))


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10):
    """
    获得一个挑选算法的所有结果
    """
    test_accuracy = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
        test_accuracy.append(np.array(read_data_then_delete(file_name, delete=False)))

    return test_accuracy


def read_data_then_delete(file_name, delete=False):
    # 构造路径
    file_path = "../results/" + file_name + ".h5"

    # 读取h5文件中rs_test_acc（result_test_accuracy）数据
    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    # 如果选择删除
    if delete:
        # 删除文件
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc
