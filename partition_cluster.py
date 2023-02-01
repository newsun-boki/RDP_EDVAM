import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
from traclus_dbscan import BestAvailableClusterCandidateIndex, TrajectoryLineSegmentFactory, TrajectoryClusterFactory, \
    dbscan
from geometry import Point
# 轨迹分段
# 采用MDL原则把，通过垂直距离与角度距离定义MDL(par)和MDL(no_par), 如果MDL(par)大于MDL(no_par)，则在当前点的
# 前一个点进行切分
from line_segment_averaging import get_representative_line_from_trajectory_line_segments
import math
from distance_functions import angular_distance, perpendicular_distance
from geometry import LineSegment

DISTANCE_OFFSET = 0.0000000001



def get_representative_lines_from_trajectory(clusters, min_num_trajectories_in_cluster, min_vertical_lines,
                                             min_prev_dist):
    rep_lines = []
    for traj_cluster in clusters:
        if traj_cluster.num_trajectories_contained() >= min_num_trajectories_in_cluster:
            rep_line = get_representative_line_from_trajectory_line_segments(
                trajectory_line_segments=traj_cluster.get_trajectory_line_segments(),
                min_vertical_lines=min_vertical_lines,
                min_prev_dist=min_prev_dist)
            rep_lines.append(rep_line)
    return rep_lines

# 执行全部步骤，返回分段结果，聚类结果和代表轨迹
def get_rep_line_result(point_iterable_list, epsilon, min_neighbors, min_num_trajectories_in_cluster,
                        min_vertical_lines,
                        min_prev_dist):
    partitioning_result, clusters = get_clustering_result(point_iterable_list=point_iterable_list, epsilon=epsilon,
                                                          min_neighbors=min_neighbors)
    partitioning_result_origin = copy.deepcopy(partitioning_result)
    clusters_origin = copy.deepcopy(clusters)

    rep_lines = get_representative_lines_from_trajectory(clusters=clusters,
                                                         min_num_trajectories_in_cluster=min_num_trajectories_in_cluster,
                                                         min_vertical_lines=min_vertical_lines,
                                                         min_prev_dist=min_prev_dist)
    return partitioning_result_origin, clusters_origin, rep_lines

def get_cluster_iterable_from_all_points_iterable(cluster_candidates, epsilon, min_neighbors):
    # dbscan()
    line_seg_index = BestAvailableClusterCandidateIndex(candidates=cluster_candidates, epsilon=epsilon)
    clusters = dbscan(cluster_candidates_index=line_seg_index, min_neighbors=min_neighbors,
                      cluster_factory=TrajectoryClusterFactory())
    return clusters

# 单独执行轨迹聚类，返回分段结果和聚类结果（簇集合）
def get_clustering_result(point_iterable_list, epsilon, min_neighbors):
    partitioning_result = get_partitioning_result(point_iterable_list)
    clusters = get_cluster_iterable_from_all_points_iterable(cluster_candidates=partitioning_result, epsilon=epsilon,
                                                             min_neighbors=min_neighbors)
    return partitioning_result, clusters

def with_spikes_removed(trajectory):
    if len(trajectory) <= 2:
        return trajectory[:]

    spikes_removed = [trajectory[0]]
    cur_index = 1
    while cur_index < len(trajectory) - 1:
        if trajectory[cur_index - 1].distance_to(trajectory[cur_index + 1]) > 0.0:
            spikes_removed.append(trajectory[cur_index])
        cur_index += 1
    spikes_removed.append(trajectory[cur_index])
    return spikes_removed

def clean_trajectories(point_iterable_list):
    """ 轨迹数据清理 """
    cleaned_input = []
    for traj in map(lambda l: with_spikes_removed(l), point_iterable_list):
        cleaned_traj = []
        if len(traj) > 1:
            prev = traj[0]
            cleaned_traj.append(traj[0])
            for pt in traj[1:]:
                if prev.distance_to(pt) > 0.0:
                    cleaned_traj.append(pt)
                    prev = pt
            if len(cleaned_traj) > 1:
                cleaned_input.append(cleaned_traj)
    return cleaned_input


# 单独执行轨迹分段，返回分段结果
def get_partitioning_result(point_iterable_list):
    cleaned_input = clean_trajectories(point_iterable_list=point_iterable_list)
    partitioning_result = get_all_trajectory_line_segments_iterable_from_all_points_iterable(
        point_iterable_list=cleaned_input)
    return partitioning_result

def get_one_trajectory_line_segments_from_points_iterable(point_iterable, trajectory_id):
    """ 从一组原始点 集合中获取 一条分段后的轨迹
    :param point_iterable: 点列表
    :param trajectory_id:
    :return: TrajectoryLineSegment
    """
    good_indices = call_partition_trajectory(point_iterable)
    good_point_iterable = filter_by_indices(good_indices=good_indices, vals=point_iterable)
    line_segs = consecutive_item_iterator_getter(item_iterable=good_point_iterable)

    def _create_traj_line_seg(line_seg):
        return TrajectoryLineSegmentFactory().new_trajectory_line_seg(line_segment=line_seg,
                                                                      trajectory_id=trajectory_id)

    return list(map(_create_traj_line_seg, line_segs))

# 单独执行轨迹分段，返回分段结果
def get_partitioning_result(point_iterable_list):
    cleaned_input = clean_trajectories(point_iterable_list=point_iterable_list)
    partitioning_result = get_all_trajectory_line_segments_iterable_from_all_points_iterable(
        point_iterable_list=cleaned_input)
    return partitioning_result

# 根据下标过滤
# 猜测是从分段的点下标集合中得到轨迹点
def filter_by_indices(good_indices, vals):
    """ 从分段算法得到的下标集合中得到 对应的轨迹点集合
    :param good_indices: 下标集合
    :param vals: 原始点数据（未分段） 集合
    :return: 分段后的点集合
    """
    vals_iter = iter(vals)
    good_indices_iter = iter(good_indices)
    out_vals = []

    num_vals = 0
    for i in good_indices_iter:
        if i != 0:
            raise ValueError("the first index should be 0, but it was " + str(i))  # 起点必须为0下标
        else:
            for item in vals_iter:
                out_vals.append(item)
                break
            num_vals = 1
            break

    max_good_index = 0
    vals_cur_index = 1

    for i in good_indices_iter:
        max_good_index = i
        for item in vals_iter:
            num_vals += 1
            if vals_cur_index == i:
                vals_cur_index += 1
                out_vals.append(item)
                break
            else:
                vals_cur_index += 1
    for i in vals_iter:
        num_vals += 1
    if num_vals < 2:
        raise ValueError("list passed in is too short")
    # 分段下标集合最大下标一定是 点集合最后一个
    if max_good_index != num_vals - 1:
        raise ValueError("last index is " + str(max_good_index) + " but there were " + str(num_vals) + " vals")
    # print(max_good_index, num_vals)
    return out_vals


def consecutive_item_iterator_getter(item_iterable):
    """ 从分段的点 集合中 得到连续的线段
    :param item_iterable: 分段后的点集合
    :return: 分段后的线段集合
    """
    # get_line_segment_from_points
    out_vals = []
    iterator = iter(item_iterable)
    last_item = None
    num_items = 0
    for item in iterator:
        num_items = 1
        last_item = item
        break
    if num_items == 0:
        raise ValueError("iterator doesn't have any values")

    for item in iterator:
        num_items += 1
        line_seg = get_line_segment_from_points(last_item, item)
        out_vals.append(line_seg)
        last_item = item

    if num_items < 2:
        raise ValueError("iterator didn't have at least two items")
    return out_vals

def get_all_trajectory_line_segments_iterable_from_all_points_iterable(point_iterable_list):
    """ 执行分段， 从点集合 的集合中获取全部 分段后的 轨迹线段
    :param point_iterable_list:  轨迹线 点集合（一条轨迹） 的 集合
    :return: list [TrajectoryLineSegment...]
    """
    out = []
    cur_trajectory_id = 0
    for point_trajectory in point_iterable_list:
        line_segments = get_one_trajectory_line_segments_from_points_iterable(point_iterable=point_trajectory,
                                                                              trajectory_id=cur_trajectory_id)
        temp = 0
        for traj_seg in line_segments:
            out.append(traj_seg)
            temp += 1
        if temp <= 0:
            raise Exception()
        cur_trajectory_id += 1
    return out

def call_partition_trajectory(trajectory_point_list):
    """ 轨迹分段
    与下不同，输入为点列表
    :param trajectory_point_list: 轨迹线 点 列表
    :return:
    """
    if len(trajectory_point_list) < 2:
        raise ValueError

    # def encoding_cost_func(trajectory_line_segs, low, high, partition_line):
    #     return encoding_cost(trajectory_line_segs, low, high,
    #                          partition_line=partition_line,
    #                          angular_dist_func=angular_distance,
    #                          perpendicular_dist_func=perpendicular_distance)
    #
    # def partition_cost_func(trajectory_line_segs, low, high):
    #     return partition_cost(trajectory_line_segs, low, high,
    #                           model_cost_func=model_cost,
    #                           encoding_cost_func=encoding_cost)

    trajectory_line_segs = list(map(lambda i: LineSegment(trajectory_point_list[i], trajectory_point_list[i + 1]),
                                    range(0, len(trajectory_point_list) - 1)))
    return partition_trajectory(trajectory_line_segs=trajectory_line_segs)


def partition_trajectory(trajectory_line_segs):
    """轨迹分段
    切分依据：比较 MDL_par 和 MDL_no_par 的大小，如果 MDL_par 大于MDL_no_par 则在前一节进行切分
    :param trajectory_line_segs: 轨迹线线段列表
    :param partition_cost_func: MDL_par
    :param no_partition_cost_func: MDL_no_par
    :return: 切分点列表
    """
    if len(trajectory_line_segs) < 1:
        raise ValueError
    low = 0
    partition_points = [0]
    last_pt = trajectory_line_segs[len(trajectory_line_segs) - 1].end
    trajectory_line_segs.append(LineSegment(last_pt, last_pt))

    for high in range(2, len(trajectory_line_segs)):
        mdl_par = partition_cost(trajectory_line_segs, low, high)
        mdl_no_par = no_partition_cost(trajectory_line_segs, low, high)

        if trajectory_line_segs[high - 2].unit_vector.almost_equals(trajectory_line_segs[high - 1].unit_vector):
            continue
        elif trajectory_line_segs[high].start.almost_equals(trajectory_line_segs[low].start) or mdl_par > mdl_no_par:
            partition_points.append(high - 1)
            low = high - 1

    partition_points.append(len(trajectory_line_segs) - 1)
    return partition_points


def partition_cost(trajectory_line_segs, low, high):
    """   MDL_par   轨迹分段开销
    MDL(cost) = L(H) + L(D|H)
    :param trajectory_line_segs:
    :param low:
    :param high:
    :return:
    """
    if low >= high:
        raise IndexError
    partition_line = LineSegment(trajectory_line_segs[low].start, trajectory_line_segs[high].start)

    model_cost = model_cost_func(partition_line)
    encoding_cost = encoding_cost_func(trajectory_line_segs, low, high, partition_line)
    return model_cost + encoding_cost


def no_partition_cost(trajectory_line_segs, low, high):
    """  计算MDL_no_par
    :param trajectory_line_segs:
    :param low:
    :param high:
    :return:
    """
    if low >= high:
        raise ValueError
    total = 0.0
    for line_seg in trajectory_line_segs[low:high]:
        total += math.log(line_seg.length, 2)
    return total


def model_cost_func(partition_line):
    """ L(H): 描述压缩模型所需要的长度
    :param partition_line:
    :return:
    """
    return math.log(partition_line.length, 2)


def encoding_cost_func(trajectory_line_segs, low, high, partition_line):
    """  L(D|H): 描述利用压缩模型所编码的数据所需要的长度
    :param trajectory_line_segs:
    :param low:
    :param high:
    :param partition_line:
    :return:
    """
    total_angular = 0.0
    total_perp = 0.0
    for line_seg in trajectory_line_segs[low:high]:
        total_angular += angular_distance(partition_line, line_seg)
        total_perp += perpendicular_distance(partition_line, line_seg)

    return math.log(total_angular + DISTANCE_OFFSET, 2) + math.log(total_perp + DISTANCE_OFFSET, 2)


def get_line_segment_from_points(point_a, point_b):
    """  从两点中得到一条线段
    :param point_a: Point()
    :param point_b: Point()
    :return: LineSegment()
    """
    return LineSegment(point_a, point_b)


def get_trajectory_line_segment_iterator_adapter(iterator_getter, get_line_segment_from_points_func):
    def _func(list, low, high, get_line_segment_from_points_func=get_line_segment_from_points):
        iterator_getter(list, low, high, get_line_segment_from_points_func)

    return _func


def get_trajectory_line_segment_iterator(list, low, high, get_line_segment_from_points_func):
    """ 从list[low, high]中得到一条轨迹线
    :param list: 点列表 [Point(), ...]
    :param low:  起始位置
    :param high: 终止位置
    :param get_line_segment_from_points_func: 两点成线函数
    :return: 线段list，即轨迹线
    """
    if high <= low:
        raise Exception('high must be greater than low index')

    line_segs = []
    cur_pos = low
    while cur_pos < high:
        line_segs.append(get_line_segment_from_points_func(list[cur_pos], list[cur_pos + 1]))
        cur_pos += 1

    return line_segs

if __name__ == "__main__":
    start_csv = 1
    end_csv = 63#63 sample trajectories in total
    #hyper param, find the meaning in https://github.com/apolcyn/traclus_impl
    epsilon=0.06
    min_neighbors=5
    min_num_trajectories_in_cluster=4
    min_vertical_lines=0
    min_prev_dist= 0.01

    trajectories_point_list = []
    for i in range(start_csv,end_csv + 1,1):
        df = pd.read_csv("./rdp2_data/rdp_"+ str(i)+".csv")
        data = np.array(df[['C_x','C_z']])
        trajectory_point = []
        for j in range(data.shape[0]):
            point =Point(data[j][0],data[j][1])
            trajectory_point.append(point)
        trajectories_point_list.append(trajectory_point)
    partition_res, clusters, replines = get_rep_line_result(point_iterable_list=trajectories_point_list,
                                                                epsilon=epsilon,
                                                                min_neighbors=min_neighbors,
                                                                min_num_trajectories_in_cluster=min_num_trajectories_in_cluster,
                                                                min_vertical_lines=min_vertical_lines,
                                                                min_prev_dist=min_prev_dist)
    # print(partition_res)
    for j in range(1,63,1):
        rdp_df = pd.read_csv("./rdp2_data/rdp_"+str(j)+".csv")
        rdp_data = np.array(rdp_df[['C_x','C_z']])
        
        rdp_dataT = rdp_data.T
        plt.plot(rdp_dataT[0],rdp_dataT[1],color='b')
    plt.xlim((-10,-25))
    plt.ylim((0,-15))
    
    i = 0;
    xs = []
    ys = []
    for reps in replines:
        i = i+1
        print("line" + str(i))
        line_x=[]
        line_y= []
        for rep in reps:
            print(str(rep.x)+ " " + str(rep.y))
            line_x.append(rep.x)
            line_y.append(rep.y)
        plt.plot(line_x,line_y,color='r')
        xs.append(line_x)
        ys.append(line_y)
    plt.show()
    

            

    