def candidate_elimination(concepts, target):
    S = ["Ø"] * len(concepts[0])
    G = [["?"] * len(concepts[0])]

    for i, example in enumerate(concepts):
        if target[i] == "Yes":
            for j in range(len(example)):
                if S[j] == "Ø":
                    S[j] = example[j]
                elif S[j] != example[j]:
                    S[j] = "?"
            G = [g for g in G if all(g[k] == "?" or g[k] == example[k] for k in range(len(example)))]
        else:
            new_G = []
            for g in G:
                if all(g[k] == "?" or g[k] == example[k] for k in range(len(example))):
                    for j in range(len(example)):
                        if S[j] != "?" and S[j] != example[j]:
                            new_h = g.copy()
                            new_h[j] = S[j]
                            new_G.append(new_h)
                else:
                    new_G.append(g)
            G = new_G

        print(f"After example {i+1}")
        print("S =", S)
        print("G =", G)
        print("-" * 40)

    return S, G


concepts = [
    ["Technical", "Senior", "Excellent", "Good", "Urban"],
    ["Technical", "Junior", "Excellent", "Good", "Urban"],
    ["Non-Technical", "Junior", "Average", "Poor", "Rural"],
    ["Technical", "Senior", "Average", "Good", "Rural"],
    ["Technical", "Senior", "Excellent", "Good", "Rural"]
]

target = ["Yes", "Yes", "No", "No", "Yes"]

S_final, G_final = candidate_elimination(concepts, target)

print("Final Specific Boundary:", S_final)
print("Final General Boundary:", G_final)