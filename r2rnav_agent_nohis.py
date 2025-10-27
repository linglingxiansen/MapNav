import habitat
import numpy as np
import cv2
import ast
import torch
import sys
sys.path.append('/mnt/hpfs/baaiei/habitat/InstructNav_r2r_lf')
from utils.semantic_prediction import SemanticPredMaskRCNN
from llava_infer import LLaVANavigationInference
from huatu3 import process_semantic_map
import pickle
from torchvision import transforms
from PIL import Image
import skimage.morphology
from constants import color_palette
import utils.pose as pu
import utils.visualization as vu
import os
import shutil
from RedNet.RedNet_model import load_rednet
from constants import mp_categories_mapping
import torch
class UnTrapHelper:
    def __init__(self):
        self.total_id = 0
        self.epi_id = 0
    
    def reset(self):
        self.total_id +=1
        self.epi_id = 0
    
    def get_action(self):
        self.epi_id +=1
        if self.epi_id ==1:
            if self.total_id %2==0:
                return 2
            else:
                return 3
        else:
            if self.total_id %2==0:
                return 2
            else:
                return 3
def count_episode_data_pairs(base_path="/home/a/L3MVN/tmp/dump/objectnav_collect0/episodes/thread_0"):
    """
    Count complete data pairs in the given directory structure.
    A complete data pair consists of depth.png, rgb.png, and sem_map.png files.

    Args:
        base_path (str): Path to the thread_0 directory

    Returns:
        tuple: (total_episodes, dict of incomplete episodes if any)
    """
    # Dictionary to store file counts for each episode
    png_count = 0
    for root,_,files in os.walk(base_path):
        png_count += sum(1 for file in files if file.lower().endswith('.png'))

    return png_count

class HM3D_Objnav_Agent:
    def __init__(self,env:habitat.Env,args,rank=0):
        self.env = env
        self.rank = rank
        
        self.args = args
        self.episode_samples = 0
        # self.planner = ShortestPathFollower(env.sim,0.5,False)
        self.cnt = 1
        self.visited_vis = None
        self.visited = None
        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=Image.NEAREST)])
        
        self.sem_pred = SemanticPredMaskRCNN(args)
        self.device = args.device
        self.red_sem_pred = load_rednet(
            self.device, ckpt='RedNet/model/rednet_semmap_mp3d_40.pth', resize=True, # since we train on half-vision
        )
        self.red_sem_pred.eval()
        self.curr_loc = None
        self.last_loc = None
        if args.visualize or args.print_images:
            self.legend = cv2.imread('docs/legend.png')
            self.vis_image = None
            self.rgb_vis = None
        self.goal_name = 'test'
        self.timestep = 0
        model = args.model_dir
        self.llava_model = LLaVANavigationInference(model_dir=model)
        self.actionlist = []
        self.dir = "{}/dump/{}/episodes".format(args.dump_location,args.exp_name)
        self.poselist = []
        self.last_action = None
        self.collision_n = 0
        self.untrap = UnTrapHelper()

    def reset(self,no):
        args = self.args
        self.timestep = 0
        self.episode_no =no
        self.curr_loc = [args.map_size_cm / 100.0 / 2.0,
                         args.map_size_cm / 100.0 / 2.0, 0.]
        self.last_loc = None
        self.episode_samples += 1
        self.episode_steps = 0
        self.obs = self.env.reset()
        self.trajectory_summary = ""  
        map_shape = (args.map_size_cm // args.map_resolution,
                     args.map_size_cm // args.map_resolution)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        if args.visualize or args.print_images:
            self.vis_image = vu.init_vis_image(self.goal_name, self.legend)

        rgb = self.obs['rgb'].astype(np.uint8)
        # print(np.min(self.depth),np.max(self.depth))
        depth = (self.obs['depth'] - 0.5) / 4.5
        # print(depth.shape)
        # print(np.min(depth),np.max(depth))
        cv2.imwrite('depth.png',depth*255)
        # exit()
        # depth = depth[:, :, None]  # 将深度图像从 (224, 224) 转换为 (224, 224, 1)
        # depth = depth / 255.0
        semantic = torch.zeros((rgb.shape[0], rgb.shape[1],1))
        state = np.concatenate((rgb, depth, semantic), axis=2).transpose(2, 0, 1)

        obs = self._preprocess_obs(state)
        obs = np.expand_dims(obs, axis=0)
        obs = torch.from_numpy(obs).float().to(self.device)
        self.position = self.env.sim.get_agent_state().sensor_states['rgb'].position
        self.rotation = self.env.sim.get_agent_state().sensor_states['rgb'].rotation
        self.actionlist = []
        self.dir = "{}/dump/{}/episodes".format(args.dump_location,args.exp_name)
        self.poselist = []
        self.last_action = None
        self.collision_n = 0
        return obs,self.position,self.rotation
    def update_trajectory(self):
        self.episode_steps += 1
        self.metrics = self.env.get_metrics()
        # self.rgb_trajectory.append(cv2.cvtColor(self.obs['rgb'],cv2.COLOR_BGR2RGB))
        # self.depth_trajectory.append((self.obs['depth']/5.0 * 255.0).astype(np.uint8))
        
        self.position = self.env.sim.get_agent_state().sensor_states['rgb'].position
        self.rotation = self.env.sim.get_agent_state().sensor_states['rgb'].rotation

        cv2.imwrite("monitor-rgb.jpg",self.obs['rgb'])
        cv2.imwrite("monitor-depth.jpg",self.obs['depth']/5.0 * 255.0)
            
    
    def step(self,planner_inputs):
        args = self.args


        
        # if self.timestep<=3:

        #     action = 1
        # else:
        #     action = 0
        if not self.env.episode_over:
            _ = self._plan(planner_inputs)
            if self.args.visualize or self.args.print_images:
                self._visualize(planner_inputs)

            cv2.imwrite(f'/mnt/hpfs/baaiei/habitat/InstructNav_r2r_lf/inference/rgb{self.rank}-{args.exp_name}.png',self.obs['rgb'][:,:,::-1])
            # cv2.imwrite(f'/mnt/hpfs/baaiei/habitat/inference/depth{self.rank}-{args.exp_name}.png',self.depth)
            cv2.imwrite(f'/mnt/hpfs/baaiei/habitat/InstructNav_r2r_lf/inference/sem{self.rank}-{args.exp_name}.png',self.vis_image[50:530, 670:1150])
            labeled_image,labels,objects = process_semantic_map(f'/mnt/hpfs/baaiei/habitat/InstructNav_r2r_lf/inference/sem{self.rank}-{args.exp_name}.png', f'/mnt/hpfs/baaiei/habitat/InstructNav_r2r_lf/inference/sem_labels{self.rank}-{args.exp_name}.png')
            images= [f'/mnt/hpfs/baaiei/habitat/InstructNav_r2r_lf/inference/rgb{self.rank}-{args.exp_name}.png',f'/mnt/hpfs/baaiei/habitat/InstructNav_r2r_lf/inference/sem_labels{self.rank}-{args.exp_name}.png']
            # print(self.obs['instruction']['text'])
            # exit()
            
            his_rgb = 0
            if special_token :
                if len(objects)>0:
                    object_str = ', '.join(objects)
                    prompt = (
                        f"<image><image>\n"
                        f"<Navigation>You are an intelligent agent in an indoor navigation task. "
                        f"Your task is to follow this instruction:{self.obs['instruction']['text']}.\n"
                        f"Available observations:\n"
                        f"<rgb_image>1. RGB image showing the current egocentric view.\n</rgb_image>"
                        f"<semantic_map>2. Top-down semantic map, this map has been gradually built from all observations collected during your navigation process since the beginning.\n"
                        f"   - As shown, this semantic map including objects such as {object_str}. \n"
                        f"   - Dark gray areas represent obstacles, light gray areas indicate traversable spaces, and white areas show unexplored regions. \n"
                        f"   - The red arrow shows your current position, orientation and the red line represents your past trajectory.\n</semantic_map>"
                        f"Based on these observations, please plan your next action.\n"
                        f"Available actions:\n"
                        f"- MOVE_FORWARD: Move one step forward\n"
                        f"- TURN_LEFT: Rotate left in place\n"
                        f"- TURN_RIGHT: Rotate right in place\n"
                        f"- STOP: End navigation\n (only use when you believe you've reached the target)\n"
                        f"What is your next action?")
                else:
                    prompt = (
                        f"<image><image>\n"
                        f"<Navigation>You are an intelligent agent in an indoor navigation task. "
                        f"Your task is to follow this instruction:{self.obs['instruction']['text']}.\n"
                        f"Available observations:\n"
                        f"<rgb_image>1. RGB image showing the current egocentric view.\n</rgb_image>"
                        f"<semantic_map>2. Top-down semantic map, this map has been gradually built from all observations collected during your navigation process since the beginning.\n"
                        f"   - Dark gray areas represent obstacles, light gray areas indicate traversable spaces, and white areas show unexplored regions. \n"
                        f"   - The red arrow shows your current position, orientation and the red line represents your past trajectory.\n</semantic_map>"
                        f"Based on these observations, please plan your next action.\n"
                        f"Available actions:\n"
                        f"- MOVE_FORWARD: Move one step forward\n"
                        f"- TURN_LEFT: Rotate left in place\n"
                        f"- TURN_RIGHT: Rotate right in place\n"
                        f"- STOP: End navigation\n (only use when you believe you've reached the target)\n"
                        f"What is your next action?")
            
            # elif his_rgb and self.timestep>=1:


            else:
                objects_str = ""
                if objects:
                    objects_str = "   - As shown, this semantic map includes objects such as: " + ", ".join(objects)
                prompt = (
                    f"<image><image>\n"
                    f"You are an intelligent agent in an indoor navigation task. "
                    f"Your task is to follow this instruction:{self.obs['instruction']['text']}.\n"
                    f"Available observations:\n"
                    f"1. RGB image showing the current egocentric view.\n"
                    f"2. Top-down semantic map, this map has been gradually built from all observations collected during your navigation process since the beginning.\n"
                    # f"   - As shown, this semantic map including objects such as {object_str}. \n"
                    f"   - Dark gray areas represent obstacles, light gray areas indicate traversable spaces, and white areas show unexplored regions. \n"
                    f"   - The red arrow shows your current position, orientation and the red line represents your past trajectory.\n"
                    f"{objects_str}.\n"
                    f"Based on these observations, please plan your next action.\n"
                    f"Available actions:\n"
                    f"- MOVE_FORWARD: Move one step forward\n"
                    f"- TURN_LEFT: Rotate left in place\n"
                    f"- TURN_RIGHT: Rotate right in place\n"
                    f"- STOP: End navigation\n (only use when you believe you've reached the target)\n"
                    f"What is your next action?")
                


            action,response = self.llava_model.get_action_id(images = images,conversation_history = [],prompt=prompt)
            
            # stop
            if self.timestep>args.stop_th:
                action = 0
            # self.actionlist.append(action)
            if action==1:
                self.actionlist.append(f'Step {self.timestep}: MOVE_FORWARD\n')
            elif action==2:
                self.actionlist.append(f'Step {self.timestep}: TURN_LEFT\n')
            elif action==3:
                self.actionlist.append(f'Step {self.timestep}: TURN_RIGHT\n')

            # print(self.actionlist)
            untrap = 0
            if self.collision_n >= 6 and untrap==1:
                action =self.untrap.get_action()
            self.obs = self.env.step(action)
            self.last_action  = action
            # print('type(self.obs):', self.obs)
            # print('type(self.obs):', self.obs['rgb'].shape) # (256, 256, 3)
            # print('type(self.obs):', self.obs['depth'].shape) # (256, 256, 1)
            

            rgb = self.obs['rgb'].astype(np.uint8)
            # depth = self.obs['depth']
            depth = (self.obs['depth'] - 0.5) / 4.5
            # depth = depth[:, :, None]  # 将深度图像从 (224, 224) 转换为 (224, 224, 1)
            semantic = torch.zeros((rgb.shape[0], rgb.shape[1],1))
            state = np.concatenate((rgb, depth, semantic), axis=2).transpose(2, 0, 1)
            obs = self._preprocess_obs(state)
            obs = np.expand_dims(obs, axis=0)
            obs = torch.from_numpy(obs).float().to(self.device)
            # self.semantic = torch.zeros(1, 16, rgb.shape[0], rgb.shape[1])
            self.update_trajectory()
            self.poselist.append(planner_inputs['sensor_pose'].cpu().numpy().tolist())
            # print(self.poselist)

            # 收集数据
            # collect = 0
            if self.env.episode_over and args.collect == 1:
                count = count_episode_data_pairs(self.dir)/3

                print(count)
                if count>200000:
                    exit()
                if self.metrics['success']==1:
                    with open('{}/thread_{}/eps_{}/replay.pkl'.format(self.dir, self.rank, self.episode_no),'wb') as f:
                        data = {
                            'actionlist': self.actionlist,
                            'instruction': self.obs['instruction']['text'],
                            'poses':self.poselist
                        }
                        pickle.dump(data, f)
                    
                    # exit()
                else:
                    if self.episode_no>5000:
                        shutil.rmtree('{}/thread_{}/eps_{}'.format(self.dir,self.rank,self.episode_no))
                    # exit()
            self.timestep +=1
            poses = [self.position,self.rotation]
            # print('poses:',poses)
            done = self.env.episode_over
            return obs,self.position,self.rotation,done


    def _plan(self, planner_inputs):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        args = self.args

        self.last_loc = self.curr_loc

        # Get Map prediction
        map_pred = np.rint(planner_inputs['map_pred'])
        exp_pred = np.rint(planner_inputs['exp_pred'])

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0 / args.map_resolution - gx1),
                 int(c * 100.0 / args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        self.visited[gx1:gx2, gy1:gy2][start[0] - 0:start[0] + 1,
                                       start[1] - 0:start[1] + 1] = 1

        # if args.visualize or args.print_images:
            # Get last loc
        last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0 / args.map_resolution - gx1),
                        int(c * 100.0 / args.map_resolution - gy1)]
        last_start = pu.threshold_poses(last_start, map_pred.shape)
        self.visited_vis[gx1:gx2, gy1:gy2] = \
            vu.draw_line(last_start, start,
                            self.visited_vis[gx1:gx2, gy1:gy2])

        # Collision check
        if self.last_action == 1 and args.collision:
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2
        
            # if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
            #     self.col_width += 2
            #     if self.col_width == 7:
            #         length = 4
            #         buf = 3
            #     self.col_width = min(self.col_width, 5)
            # else:
            #     self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold:  # Collision
                self.collision_n += 1
                # width = self.col_width
            else:
                self.untrap.reset()
        return None


    def _preprocess_obs(self, obs, use_seg=True):
        args = self.args
        # print("obs: ", obs)
        # print(obs.shape)
        # exit()
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        self.rgb = rgb

        depth = obs[:, :, 3:4]
        # print(111111111)
        # print(np.min(depth),np.max(depth))
        
        semantic = obs[:,:,4:5].squeeze()
        # print("obs: ", semantic.shape)
        if args.use_gtsem:
            self.rgb_vis = rgb
            sem_seg_pred = np.zeros((rgb.shape[0], rgb.shape[1], 15 + 1))
            for i in range(16):
                sem_seg_pred[:,:,i][semantic == i+1] = 1
        else: 
            red_semantic_pred, semantic_pred = self._get_sem_pred(
                rgb.astype(np.uint8), depth, use_seg=use_seg)
            
            sem_seg_pred = np.zeros((rgb.shape[0], rgb.shape[1], 15 + 1))   
            for i in range(0, 15):
                # print(mp_categories_mapping[i])
                sem_seg_pred[:,:,i][red_semantic_pred == mp_categories_mapping[i]] = 1

            sem_seg_pred[:,:,0][semantic_pred[:,:,0] == 0] = 0
            sem_seg_pred[:,:,1][semantic_pred[:,:,1] == 0] = 0
            sem_seg_pred[:,:,3][semantic_pred[:,:,3] == 0] = 0
            sem_seg_pred[:,:,4][semantic_pred[:,:,4] == 1] = 1
            sem_seg_pred[:,:,5][semantic_pred[:,:,5] == 1] = 1

        # sem_seg_pred = self._get_sem_pred(
        #     rgb.astype(np.uint8), depth, use_seg=use_seg)

        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)
        # print(np.min(depth),np.max(depth))
        # exit()
        self.depth = depth
        ds = args.env_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred),
                               axis=2).transpose(2, 0, 1)

        return state

    def _preprocess_depth(self, depth, min_d, max_d):
        # print(depth.shape)
        # exit()
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0
        # print(np.min(depth),np.max(depth))
        # depth = min_d * 100.0 + depth * max_d * 100.0
        depth = min_d * 100.0 + depth * (max_d-min_d) * 100.0
        # depth = depth*1000.

        return depth

    def _get_sem_pred(self, rgb, depth, use_seg=True):
        if use_seg:
            image = torch.from_numpy(rgb).to(self.device).unsqueeze_(0).float()
            depth = torch.from_numpy(depth).to(self.device).unsqueeze_(0).float()
            with torch.no_grad():
                red_semantic_pred = self.red_sem_pred(image, depth)
                red_semantic_pred = red_semantic_pred.squeeze().cpu().detach().numpy()
            semantic_pred, self.rgb_vis = self.sem_pred.get_prediction(rgb)
            semantic_pred = semantic_pred.astype(np.float32)
        else:
            semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
            self.rgb_vis = rgb[:, :, ::-1]
        return red_semantic_pred, semantic_pred

    def _visualize(self, inputs):
        args = self.args
        dump_dir = "{}/dump/{}/".format(args.dump_location,
                                        args.exp_name)
        ep_dir = '{}/episodes/thread_{}/eps_{}/'.format(
            dump_dir, self.rank, self.episode_no)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)

        local_w = inputs['map_pred'].shape[0]

        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']
        map_edge = inputs['map_edge']
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']

        sem_map = inputs['sem_map_pred']

        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        sem_map += 5

        no_cat_mask = sem_map == 20
        map_mask = np.rint(map_pred) == 1
        exp_mask = np.rint(exp_pred) == 1
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1
        edge_mask = map_edge == 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1

        sem_map[vis_mask] = 3
        sem_map[edge_mask] = 3

        selem = skimage.morphology.disk(4)

        # print(sem_map.shape,np.max(sem_map),np.min(sem_map),sem_map.dtype)
        # exit()
        self.sem_map = sem_map


        color_pal = [int(x * 255.) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1],
                                      sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                 interpolation=cv2.INTER_NEAREST)
        self.vis_image[50:530, 15:655] = self.rgb_vis
        self.vis_image[50:530, 670:1150] = sem_map_vis
        
        pos = (
            (start_x * 100. / args.map_resolution - gy1)
            * 480 / map_pred.shape[0],
            (map_pred.shape[1] - start_y * 100. / args.map_resolution + gx1)
            * 480 / map_pred.shape[1],
            np.deg2rad(-start_o)
        )

        agent_arrow = vu.get_contour_points(pos, origin=(670, 50), size=10)
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        cv2.drawContours(self.vis_image, [agent_arrow], 0, color, -1)

        if args.visualize:
            # Displaying the image
            cv2.imshow("Thread {}".format(self.rank), self.vis_image)
            cv2.waitKey(1)

        if args.print_images:
            fn = '{}/episodes/thread_{}/eps_{}/{}-{}-Vis-{}.png'.format(
                dump_dir, self.rank, self.episode_no,
                self.rank, self.episode_no, self.timestep)
            
            # 保存图片
            if args.collect:
                fn = '{}/episodes/thread_{}/eps_{}/{}-rgb.png'.format(
                    dump_dir, self.rank, self.episode_no,self.timestep)
                cv2.imwrite(fn,self.rgb_vis)
                fn = '{}/episodes/thread_{}/eps_{}/{}-depth.png'.format(
                    dump_dir, self.rank, self.episode_no,self.timestep)
                cv2.imwrite(fn,self.depth)
                fn = '{}/episodes/thread_{}/eps_{}/{}-sem_map.png'.format(
                    dump_dir, self.rank, self.episode_no,self.timestep)
                cv2.imwrite(fn,self.vis_image[50:530, 670:1150])

            
            # print(fn)
            # cv2.imwrite(fn, self.vis_image)


