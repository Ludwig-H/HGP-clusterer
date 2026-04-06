import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Optional, Tuple
from uot_sinkhorn import solve_uot_sinkhorn_gpu, uot_cost_kl_gpu

class Track:
    def __init__(self, internal_id: int, semantic_class: int, det: Dict, device: str = 'cuda'):
        self.internal_id = internal_id
        self.track_id = -1
        self.semantic_class = semantic_class
        self.device = device
        
        self.age_occlusion = 0
        self.age_total = 0
        c, dim, yaw = det["centroid"], det["dimensions"], det["yaw"]
        
        self.x = np.array([c[0], c[1], c[2], 0.0, 0.0, 0.0], dtype=float)
        self.P = np.eye(6) * 1.0
        self.P[3:6, 3:6] *= 10.0  
        
        self.H = np.zeros((3, 6))
        self.H[0:3, 0:3] = np.eye(3)
        self.R = np.eye(3) * 0.1
        self.Q = np.eye(6) * 0.05
        
        self.L, self.W, self.H_dim = dim[0], dim[1], dim[2]
        self.yaw = yaw
        self.yaw_rate = 0.0
        
        self.last_points_gpu = det["points_gpu"].clone() 
        self.pred_points_gpu = self.last_points_gpu.clone()

    def predict(self, dt: float):
        self.age_occlusion += 1
        self.age_total += 1
        
        F = np.eye(6)
        F[0, 3], F[1, 4], F[2, 5] = dt, dt, dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        
        v_tensor = torch.tensor(self.x[3:6], device=self.device, dtype=torch.float32)
        total_dt = dt * self.age_occlusion
        
        theta = self.yaw_rate * total_dt
        if np.abs(theta) > 1e-4:
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            R = torch.tensor([[cos_t, -sin_t, 0],
                              [sin_t,  cos_t, 0],
                              [0,      0,     1]], device=self.device, dtype=torch.float32)
            c_tensor = torch.mean(self.last_points_gpu, dim=0)
            centered_points = self.last_points_gpu - c_tensor
            rotated_points = torch.matmul(centered_points, R.T) + c_tensor
        else:
            rotated_points = self.last_points_gpu
            
        self.pred_points_gpu = rotated_points + v_tensor * total_dt

    def update(self, det: Dict, dt: float = 0.1):
        elapsed_t = dt * max(1, self.age_occlusion)
        self.age_occlusion = 0
        self.age_total = 0
        c, dim, yaw = det["centroid"], det["dimensions"], det["yaw"]
        
        dyaw = np.arctan2(np.sin(yaw - self.yaw), np.cos(yaw - self.yaw))
        if np.abs(dyaw) > np.pi / 2:
            dyaw = dyaw - np.sign(dyaw) * np.pi
            
        measured_yaw_rate = dyaw / elapsed_t
        alpha_yaw_rate = 0.15
        self.yaw_rate = (1 - alpha_yaw_rate) * self.yaw_rate + alpha_yaw_rate * measured_yaw_rate
        
        z = np.array([c[0], c[1], c[2]])
        S = self.H @ self.P @ self.H.T + self.R
        K_gain = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K_gain @ (z - self.H @ self.x)
        self.P = (np.eye(6) - K_gain @ self.H) @ self.P
        
        alpha = 0.2
        self.L = (1 - alpha) * self.L + alpha * dim[0]
        self.W = (1 - alpha) * self.W + alpha * dim[1]
        self.H_dim = (1 - alpha) * self.H_dim + alpha * dim[2]
        
        self.yaw = yaw
        self.last_points_gpu = det["points_gpu"].clone()
        self.pred_points_gpu = self.last_points_gpu.clone()

class CoarseToFineUOTTracker:
    def __init__(self, dt: float = 0.1, max_age: int = 5, device: str = 'cuda', verbose: bool = False):
        self.tracks: List[Track] = []
        self.next_id = 1
        self.next_internal_id = 1
        self.dt = dt
        self.device = device
        self.verbose = verbose

    def predict_all(self):
        self.tracks = [tr for tr in self.tracks if getattr(tr, "age_occlusion", 0) <= 3]
        if self.verbose and len(self.tracks) > 0:
            print(f"\\n[Tracker] Prédiction : {len(self.tracks)} pistes actives extrapolées.")
        for tr in self.tracks:
            tr.predict(self.dt)

    def compute_massive_uot(self, current_points_gpu: torch.Tensor, semantic_class: int, prior: Dict) -> Tuple[Optional[np.ndarray], List[Track]]:
        active_tracks = [tr for tr in self.tracks if tr.semantic_class == semantic_class]
        M = len(active_tracks)
        N = current_points_gpu.shape[0]
        
        if M == 0 or N == 0:
            V = np.zeros((N, M), dtype=np.float32)
            return V, active_tracks
            
        pred_clouds = [tr.pred_points_gpu for tr in active_tracks]
        lengths = [cloud.shape[0] for cloud in pred_clouds]
        mega_pred_cloud = torch.cat(pred_clouds, dim=0)
        
        K_total = mega_pred_cloud.shape[0]
        
        C_matrix = torch.cdist(current_points_gpu, mega_pred_cloud, p=2)**2
        
        gate_dist = prior.get("max_speed", 20.0) * self.dt * 2.0
        C_matrix[C_matrix > gate_dist**2] = float('inf')
        
        a_f = torch.ones(N, device=self.device)
        b_f = torch.ones(K_total, device=self.device)
        
        tau_min = 1.0
        tau = tau_min + prior.get("max_speed", 20.0) * self.dt
        P_micro = solve_uot_sinkhorn_gpu(C_matrix, a_f, b_f, epsilon=0.05, tau1=tau, tau2=tau)
        
        P_micro_cpu = P_micro.cpu().numpy()
        
        V = np.zeros((N, M), dtype=np.float32)
        start_idx = 0
        for m, length in enumerate(lengths):
            V[:, m] = np.sum(P_micro_cpu[:, start_idx:start_idx+length], axis=1)
            start_idx += length
            
        if self.verbose:
             print(f"  [UOT Massif] Classe {semantic_class} : {N} pts(t) vs {K_total} pts(t-1) dans {M} pistes.")
             
        return V, active_tracks

    def step_assign(self, detections: List[Dict], active_tracks: List[Track], V: np.ndarray, semantic_class: int, prior: Dict):
        M = len(active_tracks)
        assigned_ids = [-1] * len(detections)
        
        if M > 0:
            cost_matrix = np.zeros((len(detections), M), dtype=np.float32)
        else:
            cost_matrix = np.zeros((len(detections), 1), dtype=np.float32) # Dummy for safety
            
        all_W_C = []
        
        for i, det in enumerate(detections):
            mask = det["mask_local"]
            if np.sum(mask) == 0: 
                all_W_C.append(np.zeros(M + 1))
                continue
            
            if M > 0 and V is not None:
                S_C = np.sum(V[mask], axis=0)
                W_C = np.zeros(M + 1)
                
                for m, tr in enumerate(active_tracks):
                    W_C[m] = S_C[m] / max(1, tr.last_points_gpu.shape[0])
                
                sum_W = np.sum(W_C[:-1])
                if sum_W > 1.0:
                    W_C[:-1] /= sum_W
                    W_C[-1] = 0.0
                else:
                    W_C[-1] = 1.0 - sum_W
            else:
                W_C = np.zeros(M + 1)
                W_C[-1] = 1.0
                
            all_W_C.append(W_C)
            if M > 0:
                for m in range(M):
                    cost_matrix[i, m] = -W_C[m]

        assigned_tracks_this_step = []
        assigned_track_indices = set()
        unassigned_dets = set(range(len(detections)))
        
        if M > 0:
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(cost_matrix)
            for r, c in zip(rows, cols):
                score = -cost_matrix[r, c]
                if score >= 0.1:
                    tr = active_tracks[c]
                    tr.update(detections[r], self.dt)
                    assigned_ids[r] = tr.track_id
                    assigned_tracks_this_step.append(tr)
                    assigned_track_indices.add(c)
                    unassigned_dets.remove(r)
                    if self.verbose:
                        print(f"    -> Assignation : Cluster {r} -> Track {tr.track_id} (Score: {score:.2f})")
                        
        for c, tr in enumerate(active_tracks):
            if c not in assigned_track_indices:
                assigned_tracks_this_step.append(tr)
                
        # --- PHASE 2: COLD-START REPECHAGE (Shape Matching for age_total == 1) ---
        newborn_tracks = [tr for tr in active_tracks if tr.age_total == 1 and tr not in [t for t in assigned_tracks_this_step if t.age_occlusion == 0]]
        
        if len(newborn_tracks) > 0 and len(unassigned_dets) > 0:
            unassigned_list = list(unassigned_dets)
            repechage_cost = np.full((len(unassigned_list), len(newborn_tracks)), float('inf'), dtype=np.float32)
            
            max_speed = prior.get("max_speed", 20.0)
            tau_shape = 1.0 # Strict shape matching
            
            for i_idx, r in enumerate(unassigned_list):
                det = detections[r]
                c_det = torch.tensor(det["centroid"][:3], device=self.device, dtype=torch.float32)
                p_det = det["points_gpu"]
                n_pts = p_det.shape[0]
                if n_pts == 0: continue
                
                a_f = torch.ones(n_pts, device=self.device)
                
                for j_idx, tr in enumerate(newborn_tracks):
                    c_tr = torch.tensor(tr.x[:3], device=self.device, dtype=torch.float32)
                    dist_centers = torch.norm(c_det - c_tr).item()
                    
                    if dist_centers > max_speed * self.dt * 1.5: # 50% margin for cold start
                        continue
                        
                    # Shape alignment: translate track points to det centroid
                    translation = c_det - c_tr
                    p_tr_aligned = tr.last_points_gpu + translation
                    m_pts = p_tr_aligned.shape[0]
                    if m_pts == 0: continue
                    
                    b_f = torch.ones(m_pts, device=self.device)
                    C_micro = torch.cdist(p_tr_aligned, p_det, p=2)**2
                    
                    P_micro = solve_uot_sinkhorn_gpu(C_micro, a_f, b_f, epsilon=0.05, tau1=tau_shape, tau2=tau_shape)
                    raw_score = uot_cost_kl_gpu(P_micro, C_micro, a_f, b_f, tau1=tau_shape, tau2=tau_shape)
                    score = raw_score / max(1.0, (n_pts + m_pts) / 2.0)
                    
                    if score < prior.get("match_threshold", 3.0):
                        repechage_cost[i_idx, j_idx] = score
                        
            from scipy.optimize import linear_sum_assignment
            # Replace inf with large number for Hungarian
            valid_cost = np.where(repechage_cost == float('inf'), 1e6, repechage_cost)
            rows_rep, cols_rep = linear_sum_assignment(valid_cost)
            
            for r_idx, c_idx in zip(rows_rep, cols_rep):
                if repechage_cost[r_idx, c_idx] != float('inf'):
                    r = unassigned_list[r_idx]
                    tr = newborn_tracks[c_idx]
                    
                    # Swap the track state from unassigned to assigned
                    assigned_tracks_this_step.remove(tr) # Remove the un-updated version
                    tr.update(detections[r], self.dt)
                    assigned_ids[r] = tr.track_id
                    assigned_tracks_this_step.append(tr) # Add the updated version
                    unassigned_dets.remove(r)
                    
                    if self.verbose:
                         print(f"    -> Repêchage (Cold-Start) : Cluster {r} -> Track {tr.track_id} (Shape Score: {repechage_cost[r_idx, c_idx]:.2f})")

        # --- PHASE 3: BIRTH (NEW) ---
        for r in unassigned_dets:
            if r >= len(all_W_C): continue
            W_C = all_W_C[r]
            if np.argmax(W_C) == M and W_C[M] >= 0.8:
                new_tr = Track(self.next_internal_id, semantic_class, detections[r], self.device)
                new_tr.track_id = self.next_id
                self.next_id += 1
                self.next_internal_id += 1
                assigned_ids[r] = new_tr.track_id
                assigned_tracks_this_step.append(new_tr)
                if self.verbose:
                    print(f"    -> Naissance : Cluster {r} -> Nouvelle Track {new_tr.track_id} (Score: {W_C[M]:.2f})")

        other_class_tracks = [tr for tr in self.tracks if tr.semantic_class != semantic_class]
        self.tracks = other_class_tracks + assigned_tracks_this_step
        
        return assigned_ids

def assigned_track_ids_per_frame(ids): return ids
