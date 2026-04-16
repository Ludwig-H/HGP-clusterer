import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Optional, Tuple
from uot_sinkhorn import solve_uot_sinkhorn_gpu

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
        self.Q[3:6, 3:6] = np.eye(3) * 1.0 
        
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
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            R = torch.tensor([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], device=self.device, dtype=torch.float32)
            c_tensor = torch.mean(self.last_points_gpu, dim=0)
            self.pred_points_gpu = torch.matmul(self.last_points_gpu - c_tensor, R.T) + c_tensor + v_tensor * total_dt
        else:
            self.pred_points_gpu = self.last_points_gpu + v_tensor * total_dt

    def update(self, det: Dict, dt: float = 0.1):
        self.hits += 1
        if self.hits >= 3: 
            self.state = "Confirmed"
        elapsed_t = dt * max(1, self.age_occlusion)
        self.age_occlusion = 0
        c, dim, yaw = det["centroid"], det["dimensions"], det["yaw"]
        
        dyaw = np.arctan2(np.sin(yaw - self.yaw), np.cos(yaw - self.yaw))
        if dyaw > np.pi / 2:
            dyaw -= np.pi
        elif dyaw < -np.pi / 2:
            dyaw += np.pi
        if np.abs(dyaw) > np.pi / 2: dyaw -= np.sign(dyaw) * np.pi
        self.yaw_rate = 0.75 * self.yaw_rate + 0.25 * (dyaw / elapsed_t)
        
        z = np.array([c[0], c[1], c[2]])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(6) - K @ self.H) @ self.P
        
        self.L, self.W, self.H_dim = 0.8 * self.L + 0.2 * dim[0], 0.8 * self.W + 0.2 * dim[1], 0.8 * self.H_dim + 0.2 * dim[2]
        self.yaw = yaw
        self.last_points_gpu = det["points_gpu"].clone()
        self.pred_points_gpu = self.last_points_gpu.clone()

class CoarseToFineUOTTracker2:
    def __init__(self, dt: float = 0.1, device: str = 'cuda', verbose: bool = False):
        self.tracks: List[Track] = []
        self.next_id = 1
        self.dt = dt
        self.device = device
        self.verbose = verbose

    def predict_all(self):
        alive_tracks = [tr for tr in self.tracks if (tr.state == "Confirmed" and tr.age_occlusion < 5) or (tr.state == "Unconfirmed" and tr.age_occlusion < 1)]
        self.tracks = alive_tracks
        if self.verbose and len(self.tracks) > 0:
            print(f"\\n[Tracker] Prédiction : {len(self.tracks)} pistes actives extrapolées.")
        for tr in self.tracks: tr.predict(self.dt)

    def compute_massive_uot(self, current_points_gpu: torch.Tensor, semantic_class: int, prior: Dict) -> Tuple[Optional[np.ndarray], List[Track]]:
        active_tracks = [tr for tr in self.tracks if tr.semantic_class == semantic_class and tr.state == "Confirmed"]
        M, N = len(active_tracks), current_points_gpu.shape[0]
        if M == 0 or N == 0: return np.zeros((N, M), dtype=np.float32), active_tracks
            
        pred_clouds = [tr.pred_points_gpu for tr in active_tracks]
        lengths = [c.shape[0] for c in pred_clouds]
        mega_pred_cloud = torch.cat(pred_clouds, dim=0)
        K_total = mega_pred_cloud.shape[0]
        
        C = torch.cdist(current_points_gpu, mega_pred_cloud, p=2)**2
        gate = (prior.get("max_speed", 20.0) * self.dt * 2.0)**2
        C[C > gate] = float('inf')
        
        tau = 1.0 + prior.get("max_speed", 20.0) * self.dt
        P_micro = solve_uot_sinkhorn_gpu(C, torch.ones(N, device=self.device), torch.ones(mega_pred_cloud.shape[0], device=self.device), epsilon=0.05, tau1=tau, tau2=tau)
        
        P_cpu = P_micro.cpu().numpy()
        V = np.zeros((N, M), dtype=np.float32)
        idx = 0
        for m, l in enumerate(lengths):
            V[:, m] = np.sum(P_cpu[:, idx:idx+l], axis=1)
            idx += l
            
        if self.verbose:
             print(f"  [UOT Massif] Classe {semantic_class} : {N} pts(t) vs {K_total} pts(t-1) dans {M} pistes.")
             
        return V, active_tracks

    def step(self, detections: List[Dict], semantic_class: int, prior: Dict) -> List[int]:
        if len(detections) == 0: return []
        
        current_points = torch.cat([d["points_gpu"] for d in detections]) if detections else torch.empty(0, 3, device=self.device)
        V, confirmed_tracks = self.compute_massive_uot(current_points, semantic_class, prior)
        
        M_conf = len(confirmed_tracks)
        assigned_ids = [-1] * len(detections)
        unassigned_dets = set(range(len(detections)))
        assigned_track_indices = set()
        
        if M_conf > 0:
            cost_matrix = np.zeros((len(detections), M_conf), dtype=np.float32)
            offset = 0
            for i, det in enumerate(detections):
                n_points = det["points_gpu"].shape[0]
                if n_points == 0: continue
                S_C = np.sum(V[offset : offset + n_points], axis=0)
                offset += n_points
                W_C = S_C / np.array([max(1, tr.last_points_gpu.shape[0]) for tr in confirmed_tracks])
                sum_W = np.sum(W_C)
                if sum_W > 1.0: W_C /= sum_W
                cost_matrix[i, :] = -W_C
                
            rows, cols = linear_sum_assignment(cost_matrix)
            for r, c in zip(rows, cols):
                score = -cost_matrix[r, c]
                if score >= 0.1:
                    tr = confirmed_tracks[c]
                    tr.update(detections[r], self.dt)
                    assigned_ids[r] = tr.track_id
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

        updated_tracks = [tr for i, tr in enumerate(confirmed_tracks) if i in assigned_track_indices]
        orphan_tracks = [tr for i, tr in enumerate(confirmed_tracks) if i not in assigned_track_indices]
        
        unconfirmed = [tr for tr in self.tracks if tr.semantic_class == semantic_class and tr.state == "Unconfirmed"]
        if unassigned_dets and unconfirmed:
            U_dets = list(unassigned_dets)
            cost_unconf = np.zeros((len(U_dets), len(unconfirmed)), dtype=np.float32)
            max_dist = prior.get("max_speed", 20.0) * self.dt * 1.5
            
            for i, r in enumerate(U_dets):
                c_det = detections[r]["centroid"][:3]
                for j, tr in enumerate(unconfirmed):
                    dist = np.linalg.norm(c_det - tr.x[:3])
                    cost_unconf[i, j] = dist if dist < max_dist else 1e6
                    
            r_u, c_u = linear_sum_assignment(cost_unconf)
            for r_idx, c_idx in zip(r_u, c_u):
                if cost_unconf[r_idx, c_idx] < max_dist:
                    r = U_dets[r_idx]
                    tr = unconfirmed[c_idx]
                    if r in unassigned_dets:
                        tr.update(detections[r], self.dt)
                        assigned_ids[r] = tr.track_id
                        updated_tracks.append(tr)
                        unassigned_dets.remove(r)
                        if self.verbose:
                             print(f"    -> Repêchage (Unconfirmed) : Cluster {r} -> Track {tr.track_id} (Dist: {cost_unconf[r_idx, c_idx]:.2f})")
                else:
                    if self.verbose and cost_unconf[r_idx, c_idx] != 1e6:
                        r = U_dets[r_idx]
                        tr = unconfirmed[c_idx]
                        print(f"    -> Rejet (Unconfirmed) : Cluster {r} -> Track {tr.track_id} (Dist: {cost_unconf[r_idx, c_idx]:.2f} >= Max: {max_dist:.2f})")
            
            if self.verbose:
                assigned_rows_set = set(r_u)
                for i_idx, r in enumerate(U_dets):
                    if i_idx not in assigned_rows_set and r in unassigned_dets:
                        if np.any(cost_unconf[i_idx] != 1e6):
                            print(f"    -> Orphelin (Unconfirmed) : Cluster {r} ignoré par l'algorithme Hongrois de repêchage")
                        
        orphan_tracks.extend([tr for tr in unconfirmed if tr not in updated_tracks])
        
        for r in list(unassigned_dets):
            tr = Track(self.next_id, semantic_class, detections[r], self.device)
            tr.track_id = self.next_id
            self.next_id += 1
            assigned_ids[r] = tr.track_id
            updated_tracks.append(tr)
            unassigned_dets.remove(r)
            if self.verbose:
                print(f"    -> Naissance : Cluster {r} -> Nouvelle Track {tr.track_id}")
            
        other_class_tracks = [tr for tr in self.tracks if tr.semantic_class != semantic_class]
        self.tracks = other_class_tracks + updated_tracks + orphan_tracks
        
        return assigned_ids

def assigned_track_ids_per_frame(ids): return ids
