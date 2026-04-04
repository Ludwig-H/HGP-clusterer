import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Optional
from uot_sinkhorn import solve_uot_sinkhorn_gpu, uot_cost_kl_gpu

class Track:
    def __init__(self, track_id: int, semantic_class: int, det: Dict, device: str = 'cuda'):
        self.track_id = track_id
        self.semantic_class = semantic_class
        self.device = device
        
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
        self.pred_points_gpu = None
        
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
        elapsed_t = self.age_occlusion * dt
        if elapsed_t <= 0: elapsed_t = dt
        
        dyaw = np.arctan2(np.sin(yaw - self.yaw), np.cos(yaw - self.yaw))
        
        # Heuristique Anti-Saut (PCA ambiguity)
        # Si la rotation semble > 90°, c'est la PCA qui a inversé ses axes. On rabat.
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
        self.age_occlusion = 0
        self.hits += 1

class CoarseToFineUOTTracker:
    def __init__(self, dt: float = 0.1, max_age: int = 5, device: str = 'cuda', verbose: bool = False):
        self.tracks: List[Track] = []
        self.next_id = 1
        self.dt = dt
        self.max_age = max_age
        self.device = device
        self.verbose = verbose

    def predict_all(self):
        if self.verbose and len(self.tracks) > 0:
            print(f"\n[Tracker] Prédiction : {len(self.tracks)} pistes actives extrapolées.")
        for tr in self.tracks:
            tr.predict(self.dt)

    def _match_stage(self, tracks_subset: List[Track], det_indices: List[int], detections: List[Dict], prior: Dict, stage: int = 1):
        N = len(tracks_subset)
        M = len(det_indices)
        if N == 0 or M == 0:
            return [], list(range(N)), det_indices
            
        obs_mu = torch.tensor([detections[j]["centroid"][:2] for j in det_indices], device=self.device, dtype=torch.float32)
        if obs_mu.dim() == 1: obs_mu = obs_mu.unsqueeze(0)
        
        pred_mu = torch.tensor([tr.x[:2] for tr in tracks_subset], device=self.device, dtype=torch.float32)
        if pred_mu.dim() == 1: pred_mu = pred_mu.unsqueeze(0)
        
        # Coarse Gating on 2D BEV plane (XY) to avoid vertical bounding box noise
        C_macro = torch.cdist(pred_mu, obs_mu, p=2)**2
        
        # Scaling gate distance dynamically with occlusion age to prevent losing tracking during occlusions
        ages = torch.tensor([max(1, tr.age_occlusion) for tr in tracks_subset], device=self.device, dtype=torch.float32).unsqueeze(1)
        gate_dists = torch.clamp(prior.get("max_speed", 20.0) * self.dt * 1.5 * ages, min=2.0)
        mask_gated = C_macro > (gate_dists**2)
        C_macro[mask_gated] = float('inf')
        
        if self.verbose:
            n_gated = torch.sum(mask_gated).item()
            print(f"    - Stage {stage} Coarse Gating : {n_gated}/{N*M} paires éliminées (v > {prior.get('max_speed', 20.0)} m/s)")
            
        C_final = np.full((N, M), 1e6)
        pairs = torch.where(~mask_gated)
        
        if self.verbose:
            print(f"    - Stage {stage} Fine Matching : Analyse géométrique de {len(pairs[0])} paires potentielles...")
            
        for i, j in zip(pairs[0].tolist(), pairs[1].tolist()):
            tr = tracks_subset[i]
            det_idx = det_indices[j]
            det = detections[det_idx]
            
            p_tr = tr.pred_points_gpu
            p_det = det["points_gpu"]
            C_micro = torch.cdist(p_tr, p_det, p=2)**2
            
            n_pts, m_pts = p_tr.shape[0], p_det.shape[0]
            a_f = torch.ones(n_pts, device=self.device)
            b_f = torch.ones(m_pts, device=self.device)
            
            tau = prior.get("tau", 0.5)
            P_micro = solve_uot_sinkhorn_gpu(C_micro, a_f, b_f, epsilon=0.05, tau1=tau, tau2=tau)
            raw_score = uot_cost_kl_gpu(P_micro, C_micro, a_f, b_f, tau1=tau, tau2=tau)
            
            score = raw_score / ((n_pts + m_pts) / 2.0)
            C_final[i, j] = score
            
            match_threshold = prior.get("match_threshold", 1.0)
            if self.verbose and score < match_threshold * 2.0:
                 print(f"      * Track {tr.track_id} <-> Det {det_idx}: Score UOT = {score:.4f} ({n_pts} pts vs {m_pts} pts)")
                 
        rows, cols = linear_sum_assignment(C_final)
        
        matches = []
        unmatched_tracks = set(range(N))
        unmatched_dets = set(det_indices)
        
        match_threshold = prior.get("match_threshold", 1.0)
        
        for r, c in zip(rows, cols):
            if C_final[r, c] < match_threshold:
                det_idx = det_indices[c]
                matches.append((r, det_idx))
                unmatched_tracks.remove(r)
                unmatched_dets.remove(det_idx)
                if self.verbose:
                    print(f"    => MATCH VALIDE (Stage {stage}) : Track {tracks_subset[r].track_id} assignée à Det {det_idx} (Score: {C_final[r,c]:.4f})")
                    
        return matches, list(unmatched_tracks), list(unmatched_dets)

    def step(self, detections: List[Dict], semantic_class: int, prior: Dict):
        cl_tracks = [tr for tr in self.tracks if tr.semantic_class == semantic_class]
        
        N, M = len(cl_tracks), len(detections)
        assigned_ids = [-1] * M
        
        if self.verbose:
            print(f"\n  [Classe {semantic_class}] Association: {N} pistes existantes vs {M} nouvelles détections.")

        if N == 0:
            if self.verbose and M > 0: print(f"  [Classe {semantic_class}] Initialisation de {M} nouvelles pistes.")
            for j, det in enumerate(detections):
                new_tr = Track(self.next_id, semantic_class, det, self.device)
                self.next_id += 1
                self.tracks.append(new_tr)
                assigned_ids[j] = new_tr.track_id
            return assigned_ids
            
        if M == 0:
            if self.verbose: print(f"  [Classe {semantic_class}] Résumé: 0 matches, 0 naissances, {N} disparitions potentielles.")
            return assigned_ids

        confirmed_tracks = [tr for tr in cl_tracks if tr.hits >= 2]
        unconfirmed_tracks = [tr for tr in cl_tracks if tr.hits < 2]
        
        det_indices = list(range(M))
        
        # --- Stage 1: Pistes Confirmées (Priorité VIP) ---
        matches_conf, _, unmatch_det_1 = self._match_stage(confirmed_tracks, det_indices, detections, prior, stage=1)
        for r, det_idx in matches_conf:
            tr = confirmed_tracks[r]
            tr.update(detections[det_idx], self.dt)
            assigned_ids[det_idx] = tr.track_id
            
        # --- Stage 2: Pistes Non-Confirmées ---
        matches_unconf, _, unmatch_det_2 = self._match_stage(unconfirmed_tracks, unmatch_det_1, detections, prior, stage=2)
        for r, det_idx in matches_unconf:
            tr = unconfirmed_tracks[r]
            tr.update(detections[det_idx], self.dt)
            assigned_ids[det_idx] = tr.track_id
            
        # --- Stage 3: Nouvelles pistes ---
        for j in unmatch_det_2:
            if self.verbose: print(f"    * NOUVELLE PISTE : Det {j} devient Track {self.next_id}")
            new_tr = Track(self.next_id, semantic_class, detections[j], self.device)
            assigned_ids[j] = new_tr.track_id
            self.next_id += 1
            self.tracks.append(new_tr)
            
        if self.verbose:
            n_matches = len(matches_conf) + len(matches_unconf)
            n_new = len(unmatch_det_2)
            n_lost = len(cl_tracks) - n_matches
            print(f"  [Classe {semantic_class}] Résumé: {n_matches} matches, {n_new} naissances (non-conf), {n_lost} non-matchés.")
            
        return assigned_ids

    def cleanup(self):
        """Supprime les pistes mortes et gère le coasting (fantômes)."""
        for tr in self.tracks:
            if tr.age_occlusion > 0:
                tr.last_points_gpu = tr.pred_points_gpu.clone()
        
        # Filtre de survie :
        # - Les pistes Confirmées (hits >= 2) ont droit au Coasting (max_age)
        # - Les pistes Non-Confirmées (hits < 2) ont 1 frame de grâce (age_occlusion <= 1)
        alive_tracks = []
        for tr in self.tracks:
            if tr.hits >= 2 and tr.age_occlusion <= self.max_age:
                alive_tracks.append(tr)
            elif tr.hits < 2 and tr.age_occlusion <= 1:
                alive_tracks.append(tr)
                
        self.tracks = alive_tracks

def assigned_track_ids_per_frame(ids): return ids