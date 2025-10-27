import habitat
import sys 
# sys.path.append('/mnt/hpfs/baaiei/habitat/InstructNav_r2r_lf')
import os
import argparse
import csv
from tqdm import tqdm
from config_utils import  r2r_config
from r2rnav_agent_nohis import HM3D_Objnav_Agent
import torch
import cv2 
import numpy as np
import quaternion
import utils.pose as pu
from model import Semantic_Mapping
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
from arguments import get_args

def get_sim_location(position,rotation):
    # """Returns x, y, o pose of the agent in the Habitat simulator."""
    x = - position[2]
    y = - position[0]
    axis = quaternion.as_euler_angles(rotation)[0]
    if (axis % (2 * np.pi)) < 0.1 or (axis %
                                        (2 * np.pi)) > 2 * np.pi - 0.1:
        o = quaternion.as_euler_angles(rotation)[1]
    else:
        o = 2 * np.pi - quaternion.as_euler_angles(rotation)[1]
    if o > np.pi:
        o -= 2 * np.pi
        
    return x, y, o

def get_pose_change(last_location,position,rotation):
    """Returns dx, dy, do pose change of the agent relative to the last
    timestep."""
    curr_sim_pose = get_sim_location(position,rotation)
    dx, dy, do = pu.get_rel_pose_change(
        curr_sim_pose, last_location)
    last_location = curr_sim_pose
    return dx, dy, do
# CUDA_VISIBLE_DEVICES=1 python objnav_benchmark.py --split val1 --eval 1 --auto_gpu_config 0 -n 1 --num_eval_episodes 400 --load pretrained_models/llm_model.pt --use_gtsem 0 --num_local_steps 10 --print_images 1 --model_dir /mnt/hpfs/baaiei/habitat/workspace/checkpoints/navigation_finetune_Llava-Onevision-qwen2-r2r_hisimage_bs2/ --exp_name his_rgb --eval_episodes 1839 --collect 0 --stop_th 300
def write_metrics(metrics,path="2hisdagger_wanzheng.csv"):
    with open(path, mode="w", newline="") as csv_file:
        fieldnames = metrics[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--eval_episodes",type=int,default=500)
#     parser.add_argument("--mapper_resolution",type=float,default=0.05)
#     parser.add_argument("--path_resolution",type=float,default=0.2)
#     parser.add_argument("--path_scale",type=int,default=5)
#     return parser.parse_known_args()[0]

if __name__ == "__main__":
    args = get_args()
    num_scenes = args.num_processes
    num_episodes = int(args.num_eval_episodes)
    device = args.device = torch.device("cuda" if args.cuda else "cpu")

    g_masks = torch.ones(num_scenes).float().to(device)
    step_masks = torch.zeros(num_scenes).float().to(device)
    episode_sem_frontier = []
    episode_sem_goal = []
    episode_loc_frontier = []
    for _ in range(args.num_processes):
        episode_sem_frontier.append([])
        episode_sem_goal.append([])
        episode_loc_frontier.append([])

    finished = np.zeros((args.num_processes))
    wait_env = np.zeros((args.num_processes))

    g_process_rewards = 0
    g_total_rewards = np.ones((num_scenes))
    g_sum_rewards = 1
    g_sum_global = 1

    stair_flag = np.zeros((num_scenes))
    clear_flag = np.zeros((num_scenes))

    # Starting environments
    torch.set_num_threads(1)
    # envs = make_vec_envs(args)
    # obs, infos = envs.reset()

    torch.set_grad_enabled(False)

    # Initialize map variables:
    # Full map consists of multiple channels containing the following:
    # 1. Obstacle Map
    # 2. Exploread Area
    # 3. Current Agent Location
    # 4. Past Agent Locations
    # 5,6,7,.. : Semantic Categories
    nc = args.num_sem_categories + 4  # num channels

    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size # 2400/5=480
    local_w = int(full_w / args.global_downscaling)
    local_h = int(full_h / args.global_downscaling)

    # Initializing full and local map
    full_map = torch.zeros(num_scenes, nc, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_scenes, nc, local_w,
                            local_h).float().to(device)

    local_ob_map = np.zeros((num_scenes, local_w,
                            local_h))

    local_ex_map = np.zeros((num_scenes, local_w,
                            local_h))

    target_edge_map = np.zeros((num_scenes, local_w,
                            local_h))
    target_point_map = np.zeros((num_scenes, local_w,
                            local_h))

    # dialate for target map
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    tv_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7))

    # Initial full and local pose
    full_pose = torch.zeros(num_scenes, 3).float().to(device)
    local_pose = torch.zeros(num_scenes, 3).float().to(device)

    # Origin of local map
    origins = np.zeros((num_scenes, 3))

    # Local Map Boundaries
    lmb = np.zeros((num_scenes, 4)).astype(int)

    # Planner pose inputs has 7 dimensions
    # 1-3 store continuous global agent location
    # 4-7 store local map boundaries
    planner_pose_inputs = np.zeros((num_scenes, 7))

    
    # object_norm_inv_perplexity = torch.tensor(np.load('data/object_norm_inv_perplexity.npy')).to(device)

    def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if args.global_downscaling > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]

    def get_frontier_boundaries(frontier_loc, frontier_sizes, map_sizes):
        loc_r, loc_c = frontier_loc
        local_w, local_h = frontier_sizes
        full_w, full_h = map_sizes

        gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
        gx2, gy2 = gx1 + local_w, gy1 + local_h
        if gx1 < 0:
            gx1, gx2 = 0, local_w
        if gx2 > full_w:
            gx1, gx2 = full_w - local_w, full_w

        if gy1 < 0:
            gy1, gy2 = 0, local_h
        if gy2 > full_h:
            gy1, gy2 = full_h - local_h, full_h
 
        return [int(gx1), int(gx2), int(gy1), int(gy2)]

    def init_map_and_pose():
        full_map.fill_(0.)
        full_pose.fill_(0.)
        full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                              (local_w, local_h),
                                              (full_w, full_h))

            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                          lmb[e][0] * args.map_resolution / 100.0, 0.]

        for e in range(num_scenes):
            local_map[e] = full_map[e, :,
                                    lmb[e, 0]:lmb[e, 1],
                                    lmb[e, 2]:lmb[e, 3]]
            local_pose[e] = full_pose[e] - \
                torch.from_numpy(origins[e]).to(device).float()

    def init_map_and_pose_for_env(e):
        full_map[e].fill_(0.)
        full_pose[e].fill_(0.)
        local_ob_map[e]=np.zeros((local_w,
                            local_h))
        local_ex_map[e]=np.zeros((local_w,
                            local_h))
        target_edge_map[e]=np.zeros((local_w,
                            local_h))
        target_point_map[e]=np.zeros((local_w,
                            local_h))

        step_masks[e]=0
        stair_flag[e] = 0
        clear_flag[e] = 0


        full_pose[e, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose[e].cpu().numpy()
        planner_pose_inputs[e, :3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

        lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                          (local_w, local_h),
                                          (full_w, full_h))

        planner_pose_inputs[e, 3:] = lmb[e]
        origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                      lmb[e][0] * args.map_resolution / 100.0, 0.]

        local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
        local_pose[e] = full_pose[e] - \
            torch.from_numpy(origins[e]).to(device).float()

    init_map_and_pose()
    sem_map_module = Semantic_Mapping(args).to(device)
    sem_map_module.eval()




    # Predict semantic map from frame 1
    # poses = [[0,0,0]]
    poses = torch.from_numpy(np.asarray(
        [[0,0,0] for env_idx in range(num_scenes)])
    ).float().to(device)
    global last_location
    last_location = [0,0,0]
    # print(poses)
    eve_angle = np.asarray([0])
    

    

    habitat_config = r2r_config(stage='val_unseen',episodes=args.eval_episodes)
    print(habitat_config)
    # exit()
    habitat_env = habitat.Env(habitat_config)
    habitat_agent = HM3D_Objnav_Agent(habitat_env,args)
    evaluation_metrics = []
    step = 0
    success = 0
    for i in tqdm(range(args.eval_episodes)):
        
        if step == 0:
            obs,position,rotation = habitat_agent.reset(i)
            last_location = get_sim_location(position,rotation)
            # print(obs.shape)
            increase_local_map, local_map, local_map_stair, local_pose = \
        sem_map_module(obs, poses, local_map, local_pose, eve_angle)

            local_map[:, 0, :, :][local_map[:, 13, :, :] > 0] = 0
            planner_inputs = [{} for e in range(num_scenes)]
            for e, p_input in enumerate(planner_inputs):
                p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
                p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
                p_input['pose_pred'] = planner_pose_inputs[e]
                # p_input['goal'] = goal_maps[e]  # global_goals[e]
                p_input['map_target'] = target_point_map[e]  # global_goals[e]
                p_input['new_goal'] = 1
                p_input['sensor_pose'] = poses
                p_input['found_goal'] = 0
                p_input['wait'] = wait_env[e] or finished[e]
                if args.visualize or args.print_images:
                    p_input['map_edge'] = target_edge_map[e]
                    local_map[e, -1, :, :] = 1e-5
                    p_input['sem_map_pred'] = local_map[e, 4:, :, :
                                                        ].argmax(0).cpu().numpy()
            # print(planner_inputs['map_pred'])
            obs,position,rotation,done = habitat_agent.step(planner_inputs[0])
            
            dx,dy,do = get_pose_change(last_location,position,rotation)
            last_location = get_sim_location(position,rotation)
            poses = torch.from_numpy(np.asarray([[dx,dy,do] for env_idx in range(num_scenes)])).float().to(device)


        else:
            obs,position,rotation = habitat_agent.reset(i)
            done = 0
            last_location = get_sim_location(position,rotation)
            init_map_and_pose_for_env(0)
        while not habitat_env.episode_over and habitat_agent.episode_steps < 495 and not done:
            increase_local_map, local_map, local_map_stair, local_pose = \
            sem_map_module(obs, poses, local_map, local_pose, eve_angle)

            locs = local_pose.cpu().numpy()
            planner_pose_inputs[:, :3] = locs + origins
            local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
            for e in range(num_scenes):
                r, c = locs[e, 1], locs[e, 0]
                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                int(c * 100.0 / args.map_resolution)]
                local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.

                # work for stairs in val
                # ------------------------------------------------------------------
                if args.eval:
                # # clear the obstacle during the stairs
                    if loc_r > local_w: loc_r = local_w-1
                    if loc_c > local_h: loc_c = local_h-1

                    if stair_flag[e]:
                        # must > 0
                        if torch.any(local_map[e, 18, :, :] > 0.5):
                            local_map[e, 0, :, :] = local_map_stair[e, 0, :, :]
                        local_map[e, 0, :, :] = local_map_stair[e, 0, :, :]
                # ------------------------------------------------------------------


            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Global Policy
            step +=1
            l_step = step % 10
            if l_step == args.num_local_steps - 1:
                print('success:',success)
                # For every global step, update the full and local maps
                for e in range(num_scenes):

                    step_masks[e]+=1

                    if wait_env[e] == 1:  # New episode
                        wait_env[e] = 0.

                    
                    full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = \
                        local_map[e]
                    full_pose[e] = local_pose[e] + \
                        torch.from_numpy(origins[e]).to(device).float()

                    locs = full_pose[e].cpu().numpy()
                    r, c = locs[1], locs[0]
                    loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                    int(c * 100.0 / args.map_resolution)]

                    lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                                    (local_w, local_h),
                                                    (full_w, full_h))

                    planner_pose_inputs[e, 3:] = lmb[e]
                    origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                                lmb[e][0] * args.map_resolution / 100.0, 0.]

                    local_map[e] = full_map[e, :,
                                            lmb[e, 0]:lmb[e, 1],
                                            lmb[e, 2]:lmb[e, 3]]
                    local_pose[e] = full_pose[e] - \
                        torch.from_numpy(origins[e]).to(device).float()
            

            planner_inputs = [{} for e in range(num_scenes)]
            for e, p_input in enumerate(planner_inputs):
                # planner_pose_inputs[e, 3:] = [0, local_w, 0, local_h]
                p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
                p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
                p_input['pose_pred'] = planner_pose_inputs[e]
                p_input['sensor_pose'] = poses
                # p_input['goal'] = local_goal_maps[e]  # global_goals[e]
                p_input['map_target'] = target_point_map[e]  # global_goals[e]
                # p_input['new_goal'] = l_step == args.num_local_steps - 1
                # p_input['found_goal'] = found_goal[e]
                p_input['wait'] = wait_env[e] or finished[e]
                if args.visualize or args.print_images:
                    p_input['map_edge'] = target_edge_map[e]
                    local_map[e, -1, :, :] = 1e-5
                    p_input['sem_map_pred'] = local_map[e, 4:, :,
                                                    :].argmax(0).cpu().numpy()
   

            # obs, fail_case, done, infos = envs.plan_act_and_preprocess(planner_inputs)

            obs,position,rotation,done = habitat_agent.step(planner_inputs[0])
            
            dx,dy,do = get_pose_change(last_location,position,rotation)
            last_location = get_sim_location(position,rotation)
            poses = torch.from_numpy(np.asarray([[dx,dy,do] for env_idx in range(num_scenes)])).float().to(device)

        # print(habitat_agent.metrics)
        # exit()
        evaluation_metrics.append({'success':habitat_agent.metrics['success'],
                                'spl':habitat_agent.metrics['spl'],
                                'distance_to_goal':habitat_agent.metrics['distance_to_goal'],
                                'orcale_success':habitat_agent.orcale_success})
        if habitat_agent.metrics['distance_to_goal']<=3:
            success +=1 
        write_metrics(evaluation_metrics)
        print(evaluation_metrics[-1])
        # exit()