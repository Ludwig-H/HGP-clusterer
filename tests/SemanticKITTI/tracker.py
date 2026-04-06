import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Optional, Tuple
from uot_sinkhorn import solve_uot_sinkhorn_gpu, uot_cost_kl_gpu

class Track:
    def __init__(self, internal_id: int, semantic_class: int, det: Dict, device: str = 'cuda'):
        self.internal_id = internal_id
        self.track_id = -1  # ID officiel assigné uniquement quand Confirmé
        self.semantic_class = semantic_class
        self.device = device
        
        self.state = "Unconfirmed"
        self.hits = 1
        
        c, dim, yaw = det["centroid"], det["dimensions"], det["yaw"]
        
        # 1. État Kalman Cinématique (6D) sur CPU : [x, y, z, vx, vy, vz]
        self.x = np.array([c[0], c[1], c[2], 0.0, 0.0, 0.0], dtype=float)
        self.P = np.eye(6) * 1.0
        self.P[3:6, 3:6] *= 10.0  
        
        self.H = np.zeros((3, 6))
        self.H[0:3, 0:3] = np.eye(3)
        self.R = np.eye(3) * 0.1
        self.Q = np.eye(6) * 0.05
        
        # 2. Attributs géométriques (Hors Kalman)
        self.L, self.W, self.H_dim = dim[0], dim[1], dim[2]
        self.yaw = yaw
        self.yaw_rate = 0.0
        
        # 3. Mémoire de nuage de points GPU
        self.last_points_gpu = det["points_gpu"].clone() 
        self.pred_points_gpu = self.last_points_gpu.clone() # Initialisé
        
        self.age_occlusion = 0

    def predict(self, dt: float):
        """Prédiction Kalman et extrapolation géométrique GPU."""
        # On sauvegarde l'ancienne position pour pivoter autour du bon centre
        old_x = self.x.copy()
        
        F = np.eye(6)
        F[0, 3], F[1, 4], F[2, 5] = dt, dt, dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        
        v_tensor = torch.tensor(self.x[3:6], device=self.device, dtype=torch.float32)
        
        # --- Rotation Géométrique GPU (Yaw Rate) ---
        theta = self.yaw_rate * dt
        if np.abs(theta) > 1e-4:
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            
            # Matrice de rotation 2D sur l'axe Z (Tensor GPU)
            R = torch.tensor([[cos_t, -sin_t, 0],
                              [sin_t,  cos_t, 0],
                              [0,      0,     1]], device=self.device, dtype=torch.float32)
            
            # On recentre le nuage sur l'origine (ANCIEN centroïde)
            c_tensor = torch.tensor(old_x[:3], device=self.device, dtype=torch.float32)
            centered_points = self.last_points_gpu - c_tensor
            
            # On pivote, puis on re-décale
            rotated_points = torch.matmul(centered_points, R.T) + c_tensor
        else:
            rotated_points = self.last_points_gpu
            
        # --- Extrapolation Linéaire ---
        self.pred_points_gpu = rotated_points + v_tensor * dt
        self.age_occlusion += 1

    def update(self, det: Dict, dt: float = 0.1):
        """Mise à jour Kalman + EMA Géométrique + Yaw Rate."""
        c, dim, yaw = det["centroid"], det["dimensions"], det["yaw"]
        
        # --- Calcul de la vitesse angulaire discrète (yaw_rate) ---
        elapsed_t = max(self.age_occlusion * dt, dt)
        
        dyaw = np.arctan2(np.sin(yaw - self.yaw), np.cos(yaw - self.yaw))
        
        # Heuristique Anti-Saut (PCA ambiguity)
        if np.abs(dyaw) > np.pi / 2:
            dyaw = dyaw - np.sign(dyaw) * np.pi
            
        measured_yaw_rate = dyaw / elapsed_t
        
        # EMA plus lente (0.15) pour lisser le bruit angulaire du LiDAR
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
        
        # Stockage de l'angle brut pour la frame suivante (pas d'EMA pour éviter les instabilités aux sauts de 180°)
        self.yaw = yaw
        
        self.last_points_gpu = det["points_gpu"].clone()
        self.pred_points_gpu = self.last_points_gpu.clone() # Synchronise
        self.age_occlusion = 0
        self.hits += 1

class CoarseToFineUOTTracker:
    def __init__(self, dt: float = 0.1, max_age: int = 5, device: str = 'cuda', verbose: bool = False):
        self.tracks: List[Track] = []
        self.next_id = 1
        self.next_internal_id = 1
        self.dt = dt
        self.max_age = max_age
        self.device = device
        self.verbose = verbose

    def predict_all(self):
        if self.verbose and len(self.tracks) > 0:
            print(f"\\n[Tracker] Prédiction : {len(self.tracks)} pistes actives extrapolées.")
        for tr in self.tracks:
            tr.predict(self.dt)

    def compute_massive_uot(self, current_points_gpu: torch.Tensor, semantic_class: int, prior: Dict) -> Tuple[Optional[np.ndarray], List[Track], List[Track]]:
        """
        Phase 1 : Calcule l'UOT massif entre tous les points de la classe (frame t) 
        et tous les points des pistes actives extrapolées (frame t-1).
        Renvoie la matrice V (N_points, M_actives + 1), les pistes actives, et les pistes fantômes.
        """
        cl_tracks = [tr for tr in self.tracks if tr.semantic_class == semantic_class]
        active_tracks = [tr for tr in cl_tracks if tr.age_occlusion == 1] # 1 car predict_all() a été appelé juste avant
        ghost_tracks = [tr for tr in cl_tracks if tr.age_occlusion > 1]
        
        M = len(active_tracks)
        N = current_points_gpu.shape[0]
        
        if M == 0 or N == 0:
            V = np.zeros((N, M + 1), dtype=np.float32)
            if N > 0: V[:, M] = 1.0 # 100% NEW
            return V, active_tracks, ghost_tracks
            
        pred_clouds = [tr.pred_points_gpu for tr in active_tracks]
        lengths = [cloud.shape[0] for cloud in pred_clouds]
        mega_pred_cloud = torch.cat(pred_clouds, dim=0)
        
        K_total = mega_pred_cloud.shape[0]
        
        C_matrix = torch.cdist(current_points_gpu, mega_pred_cloud, p=2)**2
        
        # Gating pour forcer les connexions impossibles à zéro dans K (via K = exp(-C/eps) et C=inf)
        gate_dist = prior.get("max_speed", 20.0) * self.dt * 2.0
        C_matrix[C_matrix > gate_dist**2] = float('inf')
        
        a_f = torch.ones(N, device=self.device) / N
        b_f = torch.ones(K_total, device=self.device) / K_total
        
        tau = prior.get("tau", 0.5)
        P_micro = solve_uot_sinkhorn_gpu(C_matrix, a_f, b_f, epsilon=0.05, tau1=tau, tau2=tau)
        
        P_micro_cpu = P_micro.cpu().numpy()
        
        V = np.zeros((N, M + 1), dtype=np.float32)
        start_idx = 0
        for m, length in enumerate(lengths):
            # La somme de la masse UOT venant du point i (de N) vers tous les points de la piste m
            V[:, m] = np.sum(P_micro_cpu[:, start_idx:start_idx+length], axis=1)
            start_idx += length
            
        # Un point i "distribue" au mieux la masse 1/N. S'il y a des pertes, c'est NEW.
        sum_V = np.sum(V[:, :M], axis=1)
        V[:, M] = np.maximum(0.0, (1.0 / N) - sum_V)
        
        # Normalisation pour que chaque ligne somme à 1.0 (Vecteur de probabilité d'appartenance)
        row_sums = np.sum(V, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-12
        V = V / row_sums
        
        if self.verbose:
             print(f"  [UOT Massif] Classe {semantic_class} : {N} pts(t) vs {K_total} pts(t-1) dans {M} pistes.")
             
        return V, active_tracks, ghost_tracks

    def step_assign(self, detections: List[Dict], active_tracks: List[Track], ghost_tracks: List[Track], V: np.ndarray, semantic_class: int, prior: Dict):
        """
        Phase 3 : Assignation Stratifiée à 3 Tours (Actifs -> Fantômes -> Nouveaux).
        """
        M = len(active_tracks)
        assigned_ids = [-1] * len(detections)
        unassigned_det_indices = list(range(len(detections)))
        
        min_hits = prior.get("min_hits", 3)
        
        if self.verbose:
            print(f"  [Assignation] Classe {semantic_class} : {len(detections)} clusters, {M} actifs, {len(ghost_tracks)} fantômes.")
            
        # =======================================================
        # TOUR 1 : Assignation Évidente aux Actifs (Via le Tenseur V)
        # =======================================================
        if M > 0 and V is not None and len(unassigned_det_indices) > 0:
            scores = np.zeros((len(detections), M))
            
            for i in unassigned_det_indices:
                mask_local = detections[i]["mask_local"]
                if np.sum(mask_local) == 0: continue
                
                V_cluster = V[mask_local] # (N_c, M+1)
                mean_V = np.mean(V_cluster, axis=0) # (M+1,)
                
                # Le score de la piste m pour ce cluster est sa proba moyenne.
                for m in range(M):
                    # On veut que le vote soit majoritairement pour m, et qu'il batte NEW
                    if mean_V[m] > mean_V[M] and mean_V[m] > 0.3: # Exige un minimum de 30% de certitude
                        scores[i, m] = mean_V[m]
                        
            # Algorithme Hongrois pour maximiser la somme des probabilités
            if np.max(scores) > 0:
                cost_matrix = -scores
                rows, cols = linear_sum_assignment(cost_matrix)
                
                for r, c in zip(rows, cols):
                    if scores[r, c] > 0:
                        tr = active_tracks[c]
                        tr.update(detections[r], self.dt)
                        assigned_ids[r] = tr.track_id
                        unassigned_det_indices.remove(r)
                        if self.verbose:
                            name = f"Track {tr.track_id}" if tr.state == "Confirmed" else f"Track_int {tr.internal_id}"
                            print(f"    -> TOUR 1 (Actifs) : Cluster {r} -> {name} (Confiance UOT: {scores[r, c]:.2%})")

        # =======================================================
        # TOUR 2 : Recherche des Fantômes (UOT Local Point-à-Point)
        # =======================================================
        if len(unassigned_det_indices) > 0 and len(ghost_tracks) > 0:
            for i in unassigned_det_indices.copy():
                det = detections[i]
                p_det = det["points_gpu"]
                
                best_score = float('inf')
                best_ghost_idx = -1
                
                for j, ghost_tr in enumerate(ghost_tracks):
                    p_tr = ghost_tr.pred_points_gpu
                    
                    # Coarse Gating Euclidien (Centres)
                    dist_centers = torch.norm(torch.tensor(det["centroid"][:2], device=self.device) - torch.tensor(ghost_tr.x[:2], device=self.device, dtype=torch.float32))
                    gate = prior.get("max_speed", 20.0) * self.dt * 1.5 * max(1, ghost_tr.age_occlusion)
                    if dist_centers > gate:
                        continue
                        
                    C_micro = torch.cdist(p_tr, p_det, p=2)**2
                    n_pts, m_pts = p_tr.shape[0], p_det.shape[0]
                    a_f = torch.ones(n_pts, device=self.device)
                    b_f = torch.ones(m_pts, device=self.device)
                    tau = prior.get("tau", 0.5)
                    
                    P_micro = solve_uot_sinkhorn_gpu(C_micro, a_f, b_f, epsilon=0.05, tau1=tau, tau2=tau)
                    raw_score = uot_cost_kl_gpu(P_micro, C_micro, a_f, b_f, tau1=tau, tau2=tau)
                    score = raw_score / ((n_pts + m_pts) / 2.0)
                    
                    if score < best_score:
                        best_score = score
                        best_ghost_idx = j
                        
                match_threshold = prior.get("match_threshold", 1.0)
                if best_score < match_threshold * 2.0:
                    ghost_tr = ghost_tracks[best_ghost_idx]
                    ghost_tr.update(det, self.dt)
                    assigned_ids[i] = ghost_tr.track_id
                    unassigned_det_indices.remove(i)
                    ghost_tracks.pop(best_ghost_idx) # Empêche qu'un fantôme soit repris
                    if self.verbose: 
                        name = f"Track {ghost_tr.track_id}" if ghost_tr.state == "Confirmed" else f"Track_int {ghost_tr.internal_id}"
                        print(f"    -> TOUR 2 (Fantômes) : Cluster {i} -> {name} RESSUSCITÉ (Score UOT: {best_score:.2f})")
                        
        # =======================================================
        # TOUR 3 : Naissances (Nouvelles Instances)
        # =======================================================
        for i in unassigned_det_indices:
            det = detections[i]
            new_tr = Track(self.next_internal_id, semantic_class, det, self.device)
            
            if new_tr.hits >= min_hits:
                new_tr.state = "Confirmed"
                new_tr.track_id = self.next_id
                self.next_id += 1
                
            self.next_internal_id += 1
            self.tracks.append(new_tr)
            assigned_ids[i] = new_tr.track_id
            if self.verbose: print(f"    -> TOUR 3 (Naissances) : Cluster {i} -> Nouvelle instance (Interne {new_tr.internal_id})")

        return assigned_ids

    def cleanup(self):
        """Supprime les pistes mortes et gère le coasting (fantômes)."""
        # Filtre de survie :
        # - Les pistes Confirmées ont droit au Coasting (max_age)
        # - Les pistes Non-Confirmées meurent IMMÉDIATEMENT (age_occlusion == 1)
        alive_tracks = []
        for tr in self.tracks:
            if tr.state == "Confirmed" and tr.age_occlusion <= self.max_age:
                alive_tracks.append(tr)
            elif tr.state == "Unconfirmed" and tr.age_occlusion <= 1:
                alive_tracks.append(tr)
                
        self.tracks = alive_tracks

def assigned_track_ids_per_frame(ids): return ids