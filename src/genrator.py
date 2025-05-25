import json
import os
import random
import numpy as np
import pandas as pd

PROFILE_DIR = "desease_profiles"

def load_disease_profile(disease_name):
    profile_path = os.path.join(PROFILE_DIR, f"{disease_name}.json")
    if not os.path.exists(profile_path):
        raise ValueError(f"Disease profile for '{disease_name}' not found.")
    
    with open(profile_path, 'r') as file:
        profile = json.load(file)
    
    return profile

def interpolate(start, end, steps, noise_scale=0.3):
    
    return [
        start + (end - start) * (t / (steps - 1)) + np.random.normal(0, noise_scale)
        for t in range(steps)
    ]

def generate_episode(disease_name, sequence_length=20, transition_to_sick=False):
    episode = []
    all_diseases = [f.split(".")[0] for f in os.listdir(PROFILE_DIR)]
    is_healthy = disease_name == "healthy"


    sick_name = None
    switch_point = None

    if is_healthy and transition_to_sick:
        switch_point = random.randint(sequence_length // 3, sequence_length - 5)
        sick_name = random.choice([d for d in all_diseases if d != "healthy"])
        sick_profile = load_disease_profile(sick_name)
    elif not is_healthy:
        sick_name = disease_name
        sick_profile = load_disease_profile(sick_name)
    else:
        sick_profile = None


    base_vitals = {"heart_rate": 80, "blood_pressure": 115, "oxygen_saturation": 98, "temperature": 36.8}
    base_subjective = {"breathing_quality": 0, "soreness": 0, "fatigue": 0, "mental_clarity": 3, "appetite": 3}


    if sick_profile:
        vitals_progression = {
            k: interpolate(base_vitals[k], sick_profile["vitals"][k], sequence_length)
            for k in base_vitals
        }
        subjective_progression = {
            k: interpolate(base_subjective[k], sick_profile["subjective"][k], sequence_length)
            for k in base_subjective
        }

    for t in range(sequence_length):
        if transition_to_sick and is_healthy and t < switch_point:

            current_vitals = {k: v + np.random.normal(0, 0.5) for k, v in base_vitals.items()}
            current_subjective = {k: max(0, v + np.random.normal(0, 0.5)) for k, v in base_subjective.items()}
        elif sick_profile:

            current_vitals = {k: vitals_progression[k][t] for k in base_vitals}
            current_subjective = {k: subjective_progression[k][t] for k in base_subjective}
        else:

            current_vitals = {k: v + np.random.normal(0, 0.5) for k, v in base_vitals.items()}
            current_subjective = {k: max(0, v + np.random.normal(0, 0.5)) for k, v in base_subjective.items()}


        mental_state = "bad" if random.random() < 0.1 else "normal"

        step = {
            "timestep": t,
            "vitals": {k: round(v, 2) for k, v in current_vitals.items()},
            "subjective": {k: round(v, 2) for k, v in current_subjective.items()},
            "mental_state": mental_state,
            "person_id": -1  
        }

        episode.append(step)

    label_index = all_diseases.index(sick_name) if sick_name else -1
    return episode, {"disease": sick_name if sick_name else "healthy", "label": label_index}


def main(output_dir="./output", num_patients=10000, sequence_length=20):
    os.makedirs(output_dir, exist_ok=True)

    all_diseases = [f.split(".")[0] for f in os.listdir(PROFILE_DIR)] + ["healthy"]
    all_rows = []

    for i in range(num_patients):
        disease = random.choice(all_diseases)
        transition = (disease == "healthy" and random.random() < 0.01)
        episode, meta = generate_episode(disease, sequence_length=sequence_length, transition_to_sick=transition)
        for step in episode:
            step["person_id"] = i
            step["true_disease"] = meta["disease"]
            step["label"] = meta["label"]
            all_rows.append(step)

    df = pd.DataFrame(all_rows)


    unique_ids = df["person_id"].unique()
    np.random.shuffle(unique_ids)
    train_ids = unique_ids[:int(0.7 * len(unique_ids))]
    val_ids = unique_ids[int(0.7 * len(unique_ids)):int(0.85 * len(unique_ids))]
    test_ids = unique_ids[int(0.85 * len(unique_ids)):]

    df[df["person_id"].isin(train_ids)].to_csv(os.path.join(output_dir, "train.csv"), index=False)
    df[df["person_id"].isin(val_ids)].to_csv(os.path.join(output_dir, "val.csv"), index=False)
    df[df["person_id"].isin(test_ids)].to_csv(os.path.join(output_dir, "test.csv"), index=False)


    print(f"- Train: {len(train_ids)} patients")
    print(f"- Val:   {len(val_ids)} patients")
    print(f"- Test:  {len(test_ids)} patients")

if __name__ == "__main__":
    main()