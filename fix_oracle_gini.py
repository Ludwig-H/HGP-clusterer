def test_rand_index_peeling():
    # Parent cluster: 2821 points of Car A, 16 points of Car B (Mixed cluster due to BEV bridge)
    size_A = 2821
    size_B = 16
    parent_size = size_A + size_B
    
    # The Single Linkage tree proposes to peel off 4 points of Car B
    shed_B = 4
    
    # 1. Calculate 'a' (Pairs of same GT in parent)
    a = (size_A * (size_A - 1) // 2) + (size_B * (size_B - 1) // 2)
    
    # 2. Calculate 'a_prime' (Pairs of same GT in children)
    child1_A = size_A
    child1_B = size_B - shed_B
    child2_A = 0
    child2_B = shed_B
    
    a_prime = (child1_A * (child1_A - 1) // 2) + (child1_B * (child1_B - 1) // 2) + (child2_B * (child2_B - 1) // 2)
    
    # 3. Calculate P_diff (Total pairs broken by the split)
    child1_size = child1_A + child1_B
    child2_size = child2_A + child2_B
    
    total_parent_pairs = parent_size * (parent_size - 1) // 2
    kept_pairs = (child1_size * (child1_size - 1) // 2) + (child2_size * (child2_size - 1) // 2)
    P_diff = total_parent_pairs - kept_pairs
    
    # 4. Oracle Decision
    decision = P_diff > 2 * (a - a_prime)
    
    print(f"--- PREUVE MATHÉMATIQUE DU GREEDY PEELING ---")
    print(f"Parent : {size_A} pts (Car A) + {size_B} pts (Car B)")
    print(f"Split proposé : Détacher {shed_B} pts (Car B)")
    print(f"Total paires brisées (P_diff) : {P_diff}")
    print(f"Paires de même instance brisées (a - a_prime) : {a - a_prime}")
    print(f"Condition Oracle : {P_diff} > {2 * (a - a_prime)}")
    print(f"Décision Oracle : {'ACCEPTER' if decision else 'REFUSER'}")
    
    # Explication des Vrais Négatifs vs Faux Négatifs
    TN_gained = P_diff - (a - a_prime) # Paires A-B brisées (Vrais Négatifs gagnés)
    TP_lost = a - a_prime              # Paires B-B brisées (Vrais Positifs perdus)
    
    print(f"\nPourquoi l'Oracle accepte ?")
    print(f"Vrais Négatifs gagnés (Paires A-B détruites) : {TN_gained}")
    print(f"Vrais Positifs perdus (Paires B-B détruites) : {TP_lost}")
    print(f"Puisque {TN_gained} > {TP_lost}, le Rand Index global AUGMENTE.")
    print(f"L'Oracle SACRIFIE mathématiquement la petite instance pour purifier la grosse !")

test_rand_index_peeling()
