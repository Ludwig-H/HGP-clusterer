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
        
        c, dim, yaw = det["centroid"], det["dimensions"], det["yaw"]
        
        # 1. État Kalman Cinématique (6D) sur CPU : [x, y, z, vx, vy, vz]
        # Initialisation : Vitesse initiale nulle
        self.x = np.array([c[0], c[1], c[2], 0.0, 0.0, 0.0], dtype=float)
        self.P = np.eye(6) * 1.0
        self.P[3:6, 3:6] *= 10.0  # Incertitude sur la vitesse
        
        self.H = np.zeros((3, 6))
        self.H[0:3, 0:3] = np.eye(3)
        self.R = np.eye(3) * 0.1
        self.Q = np.eye(6) * 0.05
        
        # 2. Attributs géométriques (Hors Kalman)
        self.L, self.W, self.H_dim = dim[0], dim[1], dim[2]
        self.yaw = yaw
        
        # 3. Mémoire de nuage de points GPU
        self.last_points_gpu = det["points_gpu"].clone() # Tensor (N, 3) sur CUDA
        self.pred_points_gpu = None
        
        self.age_occlusion = 0

    def predict(self, dt: float):
        """Prédiction Kalman et extrapolation géométrique GPU."""
        # --- Kalman Prediction ---
        F = np.eye(6)
        F[0, 3], F[1, 4], F[2, 5] = dt, dt, dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        
        # --- Extrapolation Géométrique GPU ---
        # On déplace le nuage mémorisé selon le vecteur vitesse estimé
        v_tensor = torch.tensor(self.x[3:6], device=self.device, dtype=torch.float32)
        self.pred_points_gpu = self.last_points_gpu + v_tensor * dt
        self.age_occlusion += 1

    def update(self, det: Dict):
        """Mise à jour Kalman + EMA Géométrique."""
        c, dim, yaw = det["centroid"], det["dimensions"], det["yaw"]
        
        # --- Kalman Update ---
        z = np.array([c[0], c[1], c[2]])
        S = self.H @ self.P @ self.H.T + self.R
        K_gain = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K_gain @ (z - self.H @ self.x)
        self.P = (np.eye(6) - K_gain @ self.H) @ self.P
        
        # --- EMA Géométrique (alpha=0.2) ---
        alpha = 0.2
        self.L = (1 - alpha) * self.L + alpha * dim[0]
        self.W = (1 - alpha) * self.W + alpha * dim[1]
        self.H_dim = (1 - alpha) * self.H_dim + alpha * dim[2]
        
        # Lissage de l'angle yaw via sin/cos
        s_y = (1 - alpha) * np.sin(self.yaw) + alpha * np.sin(yaw)
        c_y = (1 - alpha) * np.cos(self.yaw) + alpha * np.cos(yaw)
        self.yaw = np.arctan2(s_y, c_y)
        
        # --- Reset Nuage et Cycle de Vie ---
        self.last_points_gpu = det["points_gpu"].clone()
        self.age_occlusion = 0

class CoarseToFineUOTTracker:
    def __init__(self, dt: float = 0.1, max_age: int = 5, device: str = 'cuda', verbose: bool = False):
        self.tracks: List[Track] = []
        self.next_id = 1
        self.dt = dt
        self.max_age = max_age
        self.device = device
        self.verbose = verbose

    def predict_all(self):
        """Avance toutes les pistes d'une frame (Coasting par défaut)."""
        if self.verbose and len(self.tracks) > 0:
            print(f"\n[Tracker] Prédiction : {len(self.tracks)} pistes actives extrapolées.")
        for tr in self.tracks:
            tr.predict(self.dt)

    def step(self, detections: List[Dict], semantic_class: int, prior: Dict):
        """
        Cycle de tracking complet : Coarse (Centroïdes) -> Fine (Points)
        """
        cl_tracks = [tr for tr in self.tracks if tr.semantic_class == semantic_class]
        
        N, M = len(cl_tracks), len(detections)
        if self.verbose:
            print(f"\n  [Classe {semantic_class}] Association: {N} pistes existantes vs {M} nouvelles détections.")

        if N == 0:
            if self.verbose and M > 0: print(f"  [Classe {semantic_class}] Initialisation de {M} nouvelles pistes.")
            return self._spawn_new(detections, semantic_class)

        # ==========================================================
        # 1. ÉTAPE COARSE : UOT SUR CENTROÏDES
        # ==========================================================
        obs_mu = torch.tensor([det["centroid"][:3] for det in detections], device=self.device, dtype=torch.float32)
        if obs_mu.dim() == 1:
            obs_mu = obs_mu.unsqueeze(0)

        pred_mu = torch.tensor([tr.x[:3] for tr in cl_tracks], device=self.device, dtype=torch.float32)
        if pred_mu.dim() == 1:
            pred_mu = pred_mu.unsqueeze(0)

        C_macro = torch.cdist(pred_mu, obs_mu, p=2)**2

        # Minimum search radius of 2.0m to account for object centroid shifts and LiDAR sparsity
        gate_dist = max(2.0, prior["max_speed"] * self.dt * 1.5)
        gate = gate_dist**2
        mask_gated = C_macro > gate
        C_macro[mask_gated] = float('inf')
        
        if self.verbose:
            n_gated = torch.sum(mask_gated).item()
            print(f"    - Coarse Gating : {n_gated}/{N*M} paires éliminées (v > {prior['max_speed']} m/s)")
        
        a, b = torch.ones(N, device=self.device), torch.ones(M, device=self.device)

        # ==========================================================
        # 2. ÉTAPE FINE : UOT POINT-À-POINT
        # ==========================================================
        C_final = np.full((N, M), 1e6)
        # Simply use the gated mask instead of full UOT at the macro level
        pairs = torch.where(~mask_gated)        
        if self.verbose:
            print(f"    - Fine Matching : Analyse géométrique de {len(pairs[0])} paires potentielles...")

        for i, j in zip(pairs[0].tolist(), pairs[1].tolist()):
            tr, det = cl_tracks[i], detections[j]
            
            p_tr = tr.pred_points_gpu
            p_det = det["points_gpu"]
            C_micro = torch.cdist(p_tr, p_det, p=2)**2
            
            n_pts, m_pts = p_tr.shape[0], p_det.shape[0]
            a_f = torch.ones(n_pts, device=self.device) / n_pts
            b_f = torch.ones(m_pts, device=self.device) / m_pts
            
            P_micro = solve_uot_sinkhorn_gpu(C_micro, a_f, b_f, epsilon=0.05, tau1=0.5, tau2=0.5)
            score = uot_cost_kl_gpu(P_micro, C_micro, a_f, b_f, tau1=0.5, tau2=0.5)
            
            C_final[i, j] = score
            if self.verbose and score < 2.0: # Log seulement si relativement proche
                 print(f"      * Track {tr.track_id} <-> Det {j}: Score UOT = {score:.4f} ({n_pts} pts vs {m_pts} pts)")

        # ==========================================================
        # 3. ASSIGNATION FINALE ET CYCLE DE VIE
        # ==========================================================
        rows, cols = linear_sum_assignment(C_final)
        matched_tr, matched_det = set(), set()
        assigned_ids = [-1] * M
        
        for r, c in zip(rows, cols):
            if C_final[r, c] < 1.0: # Seuil géométrique de validation
                tr_id = cl_tracks[r].track_id
                if self.verbose: print(f"    => MATCH VALIDE : Track {tr_id} assignée à Det {c} (Score: {C_final[r,c]:.4f})")
                cl_tracks[r].update(detections[c])
                assigned_ids[c] = tr_id
                matched_tr.add(r)
                matched_det.add(c)
        
        # Naissances
        for j, det in enumerate(detections):
            if j not in matched_det:
                new_id = self.next_id
                if self.verbose: print(f"    * NOUVELLE PISTE : Det {j} devient Track {new_id}")
                self.tracks.append(Track(new_id, semantic_class, det, self.device))
                assigned_ids[j] = new_id
                self.next_id += 1
        
        if self.verbose:
            n_lost = len(cl_tracks) - len(matched_tr)
            print(f"  [Classe {semantic_class}] Résumé: {len(matched_tr)} matches, {len(detections)-len(matched_det)} naissances, {n_lost} disparitions temporaires.")
                
        return assigned_ids

    def _spawn_new(self, detections, semantic_class):
        ids = []
        for det in detections:
            new_id = self.next_id
            self.tracks.append(Track(new_id, semantic_class, det, self.device))
            ids.append(new_id)
            self.next_id += 1
        return ids

    def cleanup(self):
        """Supprime les pistes mortes et gère le coasting (fantômes)."""
        # Pour les pistes non matchées, on a déjà fait predict()
        # donc last_points_gpu doit être mis à jour par pred_points_gpu (fantôme)
        for tr in self.tracks:
            if tr.age_occlusion > 0:
                tr.last_points_gpu = tr.pred_points_gpu.clone()
        
        self.tracks = [tr for tr in self.tracks if tr.age_occlusion <= self.max_age]

def assigned_track_ids_per_frame(ids): return ids
