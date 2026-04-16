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
        self.hits = 1
        self.state = "Unconfirmed"
        c, dim, yaw = det["centroid"], det["dimensions"], det["yaw"]
        
        self.x = np.array([c[0], c[1], c[2], 0.0, 0.0, 0.0], dtype=float)
        self.P = np.eye(6) * 1.0
        self.P[3:6, 3:6] *= 10.0  
        
        self.H = np.zeros((3, 6))
        self.H[0:3, 0:3] = np.eye(3)
        self.R = np.eye(3) * 0.1
        self.Q = np.eye(6) * 0.05
        self.Q[3:6, 3:6] = np.eye(3) * 1.0  # High uncertainty for velocity (acceleration/braking)
        
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
        
        theta = 0.0
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
        self.hits += 1
        if self.hits >= 5:
            self.state = "Confirmed"
        elapsed_t = dt * max(1, self.age_occlusion)
        self.age_occlusion = 0
        c, dim, yaw = det["centroid"], det["dimensions"], det["yaw"]
        
        dyaw = np.arctan2(np.sin(yaw - self.yaw), np.cos(yaw - self.yaw))
        if dyaw > np.pi / 2:
            dyaw -= np.pi
        elif dyaw < -np.pi / 2:
            dyaw += np.pi
        if np.abs(dyaw) > np.pi / 2:
            dyaw = dyaw - np.sign(dyaw) * np.pi
            
        measured_yaw_rate = dyaw / elapsed_t
        alpha_yaw_rate = 0.25
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
        alive_tracks = []
        for tr in self.tracks:
            if tr.state == "Confirmed" and tr.age_occlusion < 5:
                alive_tracks.append(tr)
            elif tr.state == "Unconfirmed" and tr.age_occlusion < 1:
                alive_tracks.append(tr)
                
        self.tracks = alive_tracks
        if self.verbose and len(self.tracks) > 0:
            print(f"\\n[Tracker] Prédiction : {len(self.tracks)} pistes actives extrapolées.")
        for tr in self.tracks:
            tr.predict(self.dt)

    def compute_massive_uot(self, current_points_gpu: torch.Tensor, semantic_class: int, prior: Dict) -> Tuple[Optional[np.ndarray], List[Track]]:
        active_tracks = [tr for tr in self.tracks if tr.semantic_class == semantic_class and tr.state == "Confirmed"]
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

    def step_assign_confirmed(self, detections: List[Dict], confirmed_tracks: List[Track], V_conf: np.ndarray, semantic_class: int, prior: Dict) -> Tuple[List[int], List[Track], List[int], List[np.ndarray]]:
        M_conf = len(confirmed_tracks)
        assigned_ids = [-1] * len(detections)
        assigned_tracks_this_step = []
        unassigned_dets = set(range(len(detections)))
        assigned_track_indices = set()
        
        all_W_C = []
        if M_conf == 0:
            all_W_C = [np.array([1.0]) for _ in range(len(detections))]
            return assigned_ids, assigned_tracks_this_step, list(unassigned_dets), all_W_C
            
        cost_matrix = np.zeros((len(detections), M_conf), dtype=np.float32)

        offset = 0
        for i, det in enumerate(detections):
            n_points = det["points_gpu"].shape[0]
            if n_points == 0:
                all_W_C.append(np.zeros(M_conf + 1))
                continue

            S_C = np.sum(V_conf[offset : offset + n_points], axis=0) if V_conf is not None else np.zeros(M_conf)
            offset += n_points
            W_C = np.zeros(M_conf + 1)            
            for m, tr in enumerate(confirmed_tracks):
                W_C[m] = S_C[m] / max(1, tr.last_points_gpu.shape[0])
            
            sum_W = np.sum(W_C[:-1])
            if sum_W > 1.0:
                W_C[:-1] /= sum_W
                W_C[-1] = 0.0
            else:
                W_C[-1] = 1.0 - sum_W
                
            all_W_C.append(W_C)
            for m in range(M_conf):
                cost_matrix[i, m] = -W_C[m]
                
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(cost_matrix)
        for r, c in zip(rows, cols):
            score = -cost_matrix[r, c]
            if score >= 0.1:
                tr = confirmed_tracks[c]
                tr.update(detections[r], self.dt)
                assigned_ids[r] = tr.track_id
                assigned_tracks_this_step.append(tr)
                assigned_track_indices.add(c)
                unassigned_dets.remove(r)
                if self.verbose:
                    print(f"    -> Assignation : Cluster {r} -> Track {tr.track_id} (Score: {score:.2f})")
            else:
                if self.verbose:
                    tr = confirmed_tracks[c]
                    print(f"    -> Rejet (Confirmé) : Cluster {r} non-assigné à Track {tr.track_id} (Score: {score:.2f} < 0.10)")
                    
        if self.verbose:
            assigned_rows_set = set(rows)
            for r in range(len(detections)):
                if r not in assigned_rows_set:
                    print(f"    -> Orphelin (Confirmé) : Cluster {r} ignoré par l'algorithme Hongrois (compétition perdue contre d'autres clusters)")
                    
        for c, tr in enumerate(confirmed_tracks):
            if c not in assigned_track_indices:
                assigned_tracks_this_step.append(tr)
                
        return assigned_ids, assigned_tracks_this_step, list(unassigned_dets), all_W_C

    def step_assign_unconfirmed_and_births(self, detections: List[Dict], unassigned_dets: List[int], unconfirmed_tracks: List[Track], assigned_ids: List[int], assigned_tracks_this_step: List[Track], semantic_class: int, prior: Dict, all_W_C: List[np.ndarray]) -> List[int]:
        unassigned_set = set(unassigned_dets)
        
        # --- PHASE 2: Shape Matching for ALL Unconfirmed tracks ---
        if len(unconfirmed_tracks) > 0 and len(unassigned_set) > 0:
            unassigned_list = list(unassigned_set)
            repechage_cost = np.full((len(unassigned_list), len(unconfirmed_tracks)), float('inf'), dtype=np.float32)
            
            max_speed = prior.get("max_speed", 20.0)
            tau_shape = 1.0
            
            for i_idx, r in enumerate(unassigned_list):
                det = detections[r]
                c_det = torch.tensor(det["centroid"][:3], device=self.device, dtype=torch.float32)
                p_det = det["points_gpu"]
                n_pts = p_det.shape[0]
                if n_pts == 0: continue
                
                a_f = torch.ones(n_pts, device=self.device)
                
                for j_idx, tr in enumerate(unconfirmed_tracks):
                    c_tr = torch.tensor(tr.x[:3], device=self.device, dtype=torch.float32)
                    dist_centers = torch.norm(c_det[:2] - c_tr[:2]).item()
                    
                    if dist_centers > max_speed * self.dt * 1.5:
                        if self.verbose:
                            print(f"    -> Rejet (Gating Cinématique) : Cluster {r} trop éloigné de Track {tr.track_id} (Dist: {dist_centers:.2f}m > Max: {max_speed * self.dt * 1.5:.2f}m)")
                        continue
                        
                    translation = c_det - c_tr
                    p_tr_aligned = tr.last_points_gpu + translation
                    m_pts = p_tr_aligned.shape[0]
                    if m_pts == 0: continue
                    
                    b_f = torch.ones(m_pts, device=self.device)
                    C_micro = torch.cdist(p_det, p_tr_aligned, p=2)**2
                    
                    P_micro = solve_uot_sinkhorn_gpu(C_micro, a_f, b_f, epsilon=0.05, tau1=tau_shape, tau2=tau_shape)
                    raw_score = uot_cost_kl_gpu(P_micro, C_micro, a_f, b_f, tau1=tau_shape, tau2=tau_shape)
                    score = raw_score / max(1.0, (n_pts + m_pts) / 2.0)
                    
                    if score < prior.get("match_threshold", 3.0):
                        repechage_cost[i_idx, j_idx] = score
                    else:
                        if self.verbose:
                            print(f"    -> Rejet (Déformation) : Cluster {r} vs Track {tr.track_id} (Score UOT: {score:.2f} >= Seuil: {prior.get('match_threshold', 3.0):.2f})")
                        
            from scipy.optimize import linear_sum_assignment
            valid_cost = np.where(repechage_cost == float('inf'), 1e6, repechage_cost)
            rows_rep, cols_rep = linear_sum_assignment(valid_cost)
            
            assigned_unconfirmed_indices = set()
            for r_idx, c_idx in zip(rows_rep, cols_rep):
                if repechage_cost[r_idx, c_idx] != float('inf'):
                    r = unassigned_list[r_idx]
                    tr = unconfirmed_tracks[c_idx]
                    
                    if r in unassigned_set:
                        tr.update(detections[r], self.dt)
                        assigned_ids[r] = tr.track_id
                        assigned_tracks_this_step.append(tr)
                        assigned_unconfirmed_indices.add(c_idx)
                        unassigned_set.remove(r)
                        
                        if self.verbose:
                             print(f"    -> Repêchage (Unconfirmed) : Cluster {r} -> Track {tr.track_id} (Shape Score: {repechage_cost[r_idx, c_idx]:.2f})")
                             
            if self.verbose:
                for i_idx, r in enumerate(unassigned_list):
                    if r in unassigned_set:
                        # If it had at least one valid track but wasn't assigned, it lost the competition
                        if np.any(repechage_cost[i_idx] != float('inf')):
                            print(f"    -> Orphelin (Unconfirmed) : Cluster {r} ignoré par l'algorithme Hongrois de repêchage (compétition perdue)")
                        
            for c, tr in enumerate(unconfirmed_tracks):
                if c not in assigned_unconfirmed_indices:
                    assigned_tracks_this_step.append(tr)
        else:
            assigned_tracks_this_step.extend(unconfirmed_tracks)

        # --- PHASE 3: BIRTH (NEW) with NMS ---
        birth_candidates = []
        for r in unassigned_set:
            if r >= len(all_W_C): continue
            W_C = all_W_C[r]
            
            # W_C contains M_conf + 1 elements. The last one is the NEW probability (not assigned to any CONFIRMED track)
            new_prob = W_C[-1]
            
            if new_prob >= 0.7:
                birth_candidates.append({
                    'r': r,
                    'score': new_prob,
                    'centroid': detections[r]["centroid"][:3]
                })
            else:
                if self.verbose:
                    print(f"    -> Rejet (Nouveauté) : Cluster {r} refusé comme naissance (Score NEW: {new_prob:.2f} < 0.70)")
                
        birth_candidates.sort(key=lambda x: x['score'], reverse=True)
        min_dist = prior.get("W", 1.0)
        accepted_births = []
        
        for cand in birth_candidates:
            c1 = np.array(cand['centroid'])
            too_close = False
            for acc in accepted_births:
                c2 = np.array(acc['centroid'])
                dist = np.linalg.norm(c1 - c2)
                if dist < min_dist:
                    too_close = True
                    if self.verbose:
                        print(f"    -> Rejet (NMS Spatial) : Cluster {cand['r']} supprimé car trop proche d'une autre naissance (Dist: {dist:.2f}m < {min_dist:.2f}m)")
                    break
                    
            if not too_close:
                accepted_births.append(cand)
                r = cand['r']
                new_tr = Track(self.next_internal_id, semantic_class, detections[r], self.device)
                new_tr.track_id = self.next_id
                self.next_id += 1
                self.next_internal_id += 1
                assigned_ids[r] = new_tr.track_id
                assigned_tracks_this_step.append(new_tr)
                if self.verbose:
                    print(f"    -> Naissance : Cluster {r} -> Nouvelle Track {new_tr.track_id} (Score: {cand['score']:.2f})")

        other_class_tracks = [tr for tr in self.tracks if tr.semantic_class != semantic_class]
        self.tracks = other_class_tracks + assigned_tracks_this_step
        
        return assigned_ids

def assigned_track_ids_per_frame(ids): return ids
