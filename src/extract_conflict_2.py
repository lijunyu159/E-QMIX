import os
import traci
import sys
import numpy as np
import xml.etree.ElementTree as ET
from shapely.geometry import LineString
import pandas as pd
import sumolib



# ###############################################################################冲突预警

def is_point_between(p, start, end):
    """
    判断点 p 是否在由 start 和 end 定义的线段上，并且坐标位于它们之间。

    参数:
    p -- 要判断的点的坐标元组 (x, y)。
    start -- 线段的起始点坐标元组 (x, y)。
    end -- 线段的结束点坐标元组 (x, y)。

    返回:
    如果点在两个点之间，返回 True；否则返回 False。
    """
    # 检查点的 x 坐标是否在起始点和结束点的 x 坐标之间

    x1 = min(start[0], end[0])
    x2 = max(start[0], end[0])
    y1 = min(start[1], end[1])
    y2 = max(start[1], end[1])

    if (x1 <= p[0] <= x2) and (y1 <= p[1] <= y2):
        return True
    else:
        return False


def obtain_dis_forline(splitline1):
    sumd = 0
    for i in range(0, len(splitline1) - 1):
        sumd = sumd + np.sqrt(
            (splitline1[i][0] - splitline1[i + 1][0]) ** 2 + (splitline1[i][1] - splitline1[i + 1][1]) ** 2)
    return sumd


def split_line_withgivepoint(lines1, interpoint):
    # 在指定的点打断线段
    """
    在给定的折线 lines1 上，用一个点 interpoint 把它“掰开”成两段，并且算出这两段的长度 d1、d2。
    """

    # print(interpoint)
    splitindexid = None
    for i in range(0, len(lines1) - 1):
        if is_point_between(interpoint, lines1[i], lines1[i + 1]):
            splitindexid = i
    # print(splitindexid)
    # print(lines1[:splitindexid+1])
    if splitindexid is not None:
        if splitindexid < len(lines1) - 1:

            splitline1 = lines1[:splitindexid + 1] + [interpoint]
            splitline2 = [interpoint] + lines1[splitindexid + 1:]
            d1 = obtain_dis_forline(splitline1)
            d2 = obtain_dis_forline(splitline2)
        else:
            splitline1 = lines1
            d1 = obtain_dis_forline(splitline1)
            splitline2 = []
            d2 = None
    else:
        print("点不在线段上")
    return splitline1, d1, splitline2, d2


def gettwolineinterpoint(lines1, lines2):
    linestrings1 = LineString(lines1)
    linestrings2 = LineString(lines2)
    # 寻找交点
    intersections = linestrings1.intersection(linestrings2)
    # print(list(intersections.coords))
    try:
        interpoint = list(intersections.coords)[0]
    except:
        interpoint = None
    return interpoint

def keep_non_type(conflict_pair):
    if not conflict_pair:
        return []

    # 兼容单个 pair: ('type3.0','5.0')
    if isinstance(conflict_pair, tuple) and len(conflict_pair) == 2 \
            and not isinstance(conflict_pair[0], (list, tuple)):
        pairs = [conflict_pair]
    else:
        pairs = conflict_pair

    out = []
    for item in pairs:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            continue
        for v in item:
            if isinstance(v, str) and v.startswith("type"):
                continue
            out.append(v)
    return out

def cal_conflict(laneid2lengthdicts, turn2linksdicts, ControlledLanes, Delta_t1_thesold, Delta_t2_thesold, L, Linkinterdicts):
    ControlledLanes2fristveh = []
    ControlledLanesnew = []

    # 给出避免冲突的方式
    slow_cav_ind = []  # 需要减速的车辆索引
    fast_cav_ind = []  # 需要加速的车辆索引

    for laneid in ControlledLanes:
         vehlist=traci.lane.getLastStepVehicleIDs(laneid)
         for veh in vehlist:
            ControlledLanes2fristveh.append(veh)
            ControlledLanesnew.append(laneid)
    conflict_pairs = []
    conflict_pair_keys = set()
    cur_conflict_type = 'no_collision'
    conflict_type = ['arrive_collision', 'leave_arrive_collision']

    # 分析这些车的路径
    for veh1, laneid1 in zip(ControlledLanes2fristveh, ControlledLanesnew):
        veh1route = traci.vehicle.getRoute(veh1)
        index1 = veh1route.index(laneid1[:-2])
        turndir1 = (veh1route[index1], veh1route[index1 + 1])  # 当前车辆的转向

        if len(turn2linksdicts[turndir1]) is not 1:
            sublane = int(traci.vehicle.getLaneID(veh1).split("_")[-1])
            canuselinks1 = [turn2linksdicts[turndir1][sublane]]  # 当前车辆可通过哪些连接器实现转向
        else:
            canuselinks1 = turn2linksdicts[turndir1]
        pos1 = traci.vehicle.getLanePosition(veh1)  # 当前车辆所在的车道的位置
        length_laneid1 = laneid2lengthdicts[laneid1]  # 当前车道的总长度
        speed_veh1 = traci.vehicle.getSpeed(veh1)
        veh1length = traci.vehicle.getLength(veh1)  # 当前车辆的长度
        for veh2, laneid2 in zip(ControlledLanes2fristveh, ControlledLanesnew):
            if veh1 == veh2:
                continue
            # 去重：只算一次 (veh1, veh2)
            pair_key = tuple(sorted((veh1, veh2)))
            if pair_key in conflict_pair_keys:
                continue

            veh2route = traci.vehicle.getRoute(veh2)
            index2 = veh2route.index(laneid2[:-2])
            turndir2 = (veh2route[index2], veh2route[index2 + 1])  # 当前车辆的转向
            if len(turn2linksdicts[turndir2]) is not 1:
                sublane = int(traci.vehicle.getLaneID(veh2).split("_")[-1])
                canuselinks2 = [turn2linksdicts[turndir2][sublane]]  # 当前车辆可通过哪些连接器实现转向
            else:
                canuselinks2 = turn2linksdicts[turndir2]
            pos2 = traci.vehicle.getLanePosition(veh2)  # 当前车辆所在的车道的位置
            length_laneid2 = laneid2lengthdicts[laneid2]  # 当前车道的总长度
            speed_veh2 = traci.vehicle.getSpeed(veh2)
            veh2length = traci.vehicle.getLength(veh2)  # 当前车辆的长度

            # 如果速度为0，避免除0
            if speed_veh1 <= 0.001 or speed_veh2 <= 0.001:
                continue

            found_conflict = False

            # 计算每种相冲突的场景
            for link1 in canuselinks1:
                for link2 in canuselinks2:
                    if (link1, link2) in Linkinterdicts:

                        tempdicts = Linkinterdicts[(link1, link2)]
                        remian_d_veh1 = tempdicts["d1"] + length_laneid1 - pos1 + veh1length / 2  # 车辆veh1距离冲突点的距离
                        t1 = round(remian_d_veh1 / speed_veh1, 1)  # 车辆veh1到达冲突点的时间
                        remian_d_veh2 = tempdicts["d11"] + length_laneid2 - pos2 + veh2length / 2  # 车辆veh2距离冲突点的距离
                        t2 = round(remian_d_veh2 / speed_veh2, 1)  # 车辆veh2到达冲突点的时间

                        left_conflict_d1 = tempdicts[
                                               "d1"] + length_laneid1 - pos1 + veh1length + L / 2  # 车辆veh1距离离开冲突点的距离    L为车道宽度
                        left_conflict_d2 = tempdicts[
                                               "d11"] + length_laneid2 - pos2 + veh2length + L / 2  # 车辆veh2距离离开冲突点的距离  L为车道宽度
                        t11 = round(left_conflict_d1 / speed_veh1, 1)  # 车辆veh1离开冲突点的时间
                        t22 = round(left_conflict_d2 / speed_veh2, 1)  # 车辆veh2离开冲突点的时间
                        Delta_t1 = abs(t1 - t2)  # 到达冲突点时间差

                        if t1 > t2:
                            Delta_t2 = t11 - t2  # 前车离开-后车达到冲突点时间差
                        else:
                            Delta_t2 = t22 - t1

                        if Delta_t1 < Delta_t1_thesold:  # 这里delta_t1 < 阈值，说明俩车可能会迎头相撞

                            # 判断两车是否是CAVs
                            if not ("type" in veh1 and "type" in veh2):  # 排除两车均为HDVs
                                # conflict_pairs.append((veh1, veh2))
                                conflict_pair_keys.add(pair_key)
                                # for conflict_pair in conflict_pairs:
                                # 两车肯定有一个是CAVs，找到并且判断该车是需要加速还是减速
                                conflict_vehs = keep_non_type((veh1, veh2))
                                if len(conflict_vehs) == 1:
                                    if not "type" in veh1:
                                        cav = veh1
                                        hdv_t = t2
                                        cav_t = t1
                                    else:
                                        cav = veh2
                                        hdv_t = t1
                                        cav_t = t2
                                    # CAV与HDV产生冲突，此时为迎头碰撞
                                    # 如果CAV到的慢，就让他再慢点
                                    if hdv_t < cav_t:
                                        slow_cav_ind.append(int(cav))
                                    # 如果CAV到的快，就让他再快点
                                    else:
                                        fast_cav_ind.append(int(cav))
                                if len(conflict_vehs) == 2:
                                    # 此时是两辆CAVs产生冲突,此时，t 更大的车要减速，t更小的车要加速
                                    if t1>t2:
                                        slow_cav_ind.append(int(veh1))
                                        fast_cav_ind.append(int(veh2))
                                    else:
                                        slow_cav_ind.append(int(veh2))
                                        fast_cav_ind.append(int(veh1))

                                found_conflict = True
                                # cur_conflict_type = conflict_type[0]
                                break
                        elif Delta_t2 < Delta_t2_thesold:  # 此时可能发生离开-到达冲突  快的快点，慢的慢点
                            if not ("type" in veh1 and "type" in veh2):
                                # conflict_pairs.append((veh1, veh2))
                                conflict_pair_keys.add(pair_key)
                                found_conflict = True
                                # for conflict_pair in conflict_pairs:
                                conflict_vehs = keep_non_type((veh1, veh2))
                                if len(conflict_vehs) == 1:
                                    if not "type" in veh1:
                                        cav = veh1
                                        hdv_t_arr = t2; hdv_t_leave = t22
                                        cav_t_arr = t1; cav_t_leave = t11
                                    else:
                                        cav = veh2
                                        hdv_t_arr = t1; hdv_t_leave = t11
                                        cav_t_arr = t2; cav_t_leave = t22
                                    if cav_t_arr < hdv_t_arr:
                                        fast_cav_ind.append(int(cav))
                                    # if cav_t_arr > hdv_t_arr and cav_t_leave<hdv_t_leave: # 到的晚 离开的早
                                    else:
                                        slow_cav_ind.append(int(cav))
                                if len(conflict_vehs) == 2:
                                    # try:
                                    if t1 < t2:
                                        fast_cav_ind.append(int(veh1))
                                        slow_cav_ind.append(int(veh2))
                                    else:
                                        fast_cav_ind.append(int(veh2))
                                        slow_cav_ind.append(int(veh1))
                                    # except:
                                    #     print(veh1)


                            # cur_conflict_type = conflict_type[1]
                            break
                if found_conflict:
                    break

    return fast_cav_ind, slow_cav_ind
    # if conflict_pairs:
    #     return 1, conflict_pairs
    # else:
    #     return 0, [(), 'no_collision']

# def _get_ind():
#
#     """
#
#     Args:
#         ControlledLanes2fristveh:
#         ControlledLanesnew:
#
#     Returns: 找到所有应该减速的车辆和应该加速的车辆
#
#     """
#     slow_cav_ind = [] # 需要减速的车辆索引
#     fast_cav_ind = [] # 需要加速的车辆索引
#     is_danger, conflict_pairs = safe_module() # 是否安全(0/1)，conflict_pairs 包括了冲突类型
#     if is_danger is 1:
#         # todo: 逻辑还是得改阿，，依然撞车
#         for conflict_pair_type in conflict_pairs:
#             conflict_pair = conflict_pair_type[0]
#             conflict_type = conflict_pair_type[1]
#             conflict_cav = keep_non_type(conflict_pair)  # 提取冲突对中的CAVs
#             if conflict_type is 'arrive_collision':
#                 if len(conflict_cav) == 1: # HDV和CAV产生冲突
#                     slow_cav_ind.append(int(conflict_cav[0]))
#                 if len(conflict_cav) == 2: # CAV和CAV产生冲突
#                     # cav1_pos = max(abs(traci.vehicle.getPosition(conflict_cav[0])))
#                     cav1_pos = tuple(abs(x) for x in traci.vehicle.getPosition(conflict_cav[0]))
#                     cav2_pos = tuple(abs(x) for x in traci.vehicle.getPosition(conflict_cav[1]))
#                     if 89 - max(cav1_pos) < 89 - max(cav2_pos): # 这里还应该考虑速度吧，用TTC？距离冲突点的距离/当前速度，时间短的快点走，时间长的慢点走，并且要注意不要步入下一个冲突类型
#                         fast_cav_ind.append(int(conflict_cav[0]))
#                         slow_cav_ind.append(int(conflict_cav[1]))
#                     else:
#                         fast_cav_ind.append(int(conflict_cav[1]))
#                         slow_cav_ind.append(int(conflict_cav[0]))
#             if conflict_type is 'leave_arrive_collision':
#                 if len(conflict_cav) == 1:
#                     cav_pos = tuple(abs(x) for x in traci.vehicle.getPosition(conflict_cav[0]))
#                     hdv_pos = tuple(abs(x) for x in traci.vehicle.getPosition(conflict_pair[1]))
#                     if 89 - max(cav_pos) < 89 - max(hdv_pos): # CAV距离路口更近，加加速就过去了
#                         fast_cav_ind.append(int(conflict_cav[0]))
#                     else: # HDV离路口更近，CAV要减速
#                         slow_cav_ind.append(int(conflict_cav[0]))
#                 if len(conflict_cav) == 2:
#                     cav1_pos = tuple(abs(x) for x in traci.vehicle.getPosition(conflict_cav[0]))
#                     cav2_pos = tuple(abs(x) for x in traci.vehicle.getPosition(conflict_cav[1]))
#                     if 89 - max(cav1_pos) < 89 - max(cav2_pos):
#                         fast_cav_ind.append(int(conflict_cav[0]))
#                         slow_cav_ind.append(int(conflict_cav[1]))
#                     else:
#                         fast_cav_ind.append(int(conflict_cav[1]))
#                         slow_cav_ind.append(int(conflict_cav[0]))
#



